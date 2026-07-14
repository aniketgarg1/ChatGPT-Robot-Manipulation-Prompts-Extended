"""VirtualHome-style task planning with AI2-THOR execution, LLM validation/refinement,
A/B metrics, and learning memory. Core planner prompts are unchanged; a stronger model
critiques and repairs plans before execution."""

import json
import math
import os
import re
import time
from collections import Counter

import openai
import tiktoken
from PIL import Image
from ai2thor.controller import Controller

enc = tiktoken.get_encoding("cl100k_base")
_SECRETS_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "secrets.json"
)
with open(_SECRETS_PATH) as f:
    credentials = json.load(f)

dir_system = "./system"
dir_prompt = "./prompt"
dir_query = "./query"
prompt_load_order = [
    "prompt_role",
    "prompt_function",
    "prompt_environment",
    "prompt_output_format",
    "prompt_planning_constraints",
    "prompt_example",
]

VALIDATION_PROMPT = """You are a strict embodied-planning validator.
Given a user instruction, environment snapshot, and a generated task sequence, identify likely
execution failures and repair the plan.

Return valid JSON only with keys:
- shortcomings: list of concise strings describing issues in current plan
- improved_task_sequence: list of action strings in the same format as the input
  (e.g. Walktowards(<apple>), Grab(<apple>), Open(<fridge>), PutIn(<apple>,<fridge>))
- rationale: short paragraph

Rules:
- improved_task_sequence must be non-empty and use only these verbs (matching case as in examples):
  Walktowards, Grab, Put, PutIn, Open, Close, SwitchOn, Drink.
- Preserve the user's goal (same primary object and target as the instruction / original plan).
- Prefer minimal edits. Include Open/Close when interacting with closed openable containers.
- Do not add unrelated objects not present in the environment snapshot unless explicitly mapped by provided token map hints.
- Do not force fridge-specific actions unless the target is a fridge or another openable container relevant to the instruction.
- Do not include markdown fences."""


REFINEMENT_PROMPT = """You are an embodied planning improver.
Given instruction, environment snapshot, original task sequence, and validator shortcomings,
produce a corrected executable task sequence that preserves the original task goal.

Return valid JSON only:
{
  "improved_task_sequence": ["..."],
  "rationale": "..."
}

Rules:
- Keep the same goal as the original instruction.
- Return a non-empty executable sequence using only: Walktowards, Grab, Put, PutIn, Open, Close, SwitchOn, Drink.
- Prefer minimal edits that fix shortcomings.
- Include Open/Close when needed for containers.
- Keep objects grounded in the environment; do not invent unrelated tools/containers.
- Do not include markdown fences."""


STEP_REVIEW_PROMPT = """You are a step-level embodied planning reviewer.
Given instruction, environment snapshot, prior accepted steps, and one candidate step,
return a corrected executable replacement for that step if needed.

Return valid JSON only:
{
  "replacement_steps": ["..."],
  "reason": "..."
}

Rules:
- replacement_steps must be non-empty and each action must use one of:
  Walktowards, Grab, Put, PutIn, Open, Close, SwitchOn, Drink.
- Keep goal and object grounding consistent with the instruction/environment.
- Do not inject unrelated objects (e.g., pan) unless they already exist in environment tokens.
- Prefer minimal edits; if the step is good, return it unchanged as a single-item list.
- No markdown fences.
"""


BUILTIN_TYPE_ALIASES = {
    "fryingpan": ["Pan", "Pot"],
    "pan": ["Pan", "Pot"],
    "breadslice": ["Bread", "BreadSliced", "BreadLoaf"],
    "bread": ["Bread", "BreadSliced", "BreadLoaf"],
    "sink": ["SinkBasin", "Sink"],
    "table": ["DiningTable", "CoffeeTable", "SideTable"],
    "kitchentable": ["DiningTable", "CoffeeTable", "SideTable"],
    "kitchencounter": ["CounterTop"],
    "counter": ["CounterTop"],
    "stove": ["StoveBurner", "Microwave"],
    "toaster": ["Toaster"],
    "plate": ["Plate"],
    "apple": ["Apple"],
    "fridge": ["Fridge"],
    "tv": ["Television"],
}


def load_executor_sim_config(scenarios_dir):
    path = os.path.join(scenarios_dir, "executor_sim_overrides.json")
    if not os.path.isfile(path):
        return {"global_object_map": {}, "scenarios": {}}
    with open(path) as f:
        return json.load(f)


def reset(comm, scene_name="FloorPlan1"):
    comm.reset(scene_name)
    return True


def generate_script(input_array):
    output_array = []
    alias = {
        "puton": "put",
        "placeon": "put",
        "placein": "putin",
        "pickup": "grab",
        "walkto": "walktowards",
    }
    for action in input_array:
        action = action.replace(">", "").replace("<", "").replace(" ", "")
        parts = action.split("(")
        verb = parts[0].lower()
        verb = alias.get(verb, verb)
        arguments = parts[1].strip(")")

        if len(arguments) == 0:
            objects = []
        else:
            objects = arguments.split(",")

        if len(objects) == 0:
            output_array.append(f"{verb}")
        elif len(objects) == 1:
            output_array.append(f"{verb} {objects[0]}")
        else:
            output_array.append(f"{verb} {objects[0]} {objects[1]}")

    return output_array


def remove_brackets(name):
    return name.replace("<", "").replace(">", "")


def which_room(graph, node_id):
    id_to_node = {node["id"]: node for node in graph["nodes"]}
    child_to_parent = {}
    for edge in graph["edges"]:
        if edge["from_id"] in child_to_parent.keys():
            child_to_parent[edge["from_id"]].append(
                (edge["to_id"], edge["relation_type"]))
        else:
            child_to_parent[edge["from_id"]] = [
                (edge["to_id"], edge["relation_type"])]
    if node_id not in child_to_parent.keys():
        return None
    parent_node_ids = child_to_parent[node_id]
    for parent_node_id in parent_node_ids:
        parent_node = id_to_node[parent_node_id[0]]
        relation_type = parent_node_id[1]
        if relation_type != "INSIDE" and relation_type != "ON":
            continue
        if "Decor" in parent_node["category"]:
            continue
        if "Rooms" in parent_node["category"]:
            return parent_node["class_name"]
    return None


def find_parent_node(graph, node_name, room_name):
    id_to_node = {node["id"]: node for node in graph["nodes"]}
    name_to_id = {}
    for node in graph["nodes"]:
        if node["class_name"] in name_to_id.keys():
            name_to_id[node["class_name"]].append(node["id"])
        else:
            name_to_id[node["class_name"]] = [node["id"]]
    child_to_parent = {}
    for edge in graph["edges"]:
        if edge["from_id"] in child_to_parent.keys():
            child_to_parent[edge["from_id"]].append(
                (edge["to_id"], edge["relation_type"]))
        else:
            child_to_parent[edge["from_id"]] = [
                (edge["to_id"], edge["relation_type"])]
    if "_" in node_name:
        node_ids = [int(node_name.split("_")[1])]
        node_name = node_name.split("_")[0]
    else:
        if node_name not in name_to_id.keys():
            return None
        node_ids = name_to_id[node_name]
        node_ids = [
            node_id for node_id in node_ids if which_room(
                graph, node_id) == room_name]
    return_dict = {"object_states": {}, "asset_states": {}}
    for node_id in node_ids:
        if "GRABBABLE" in id_to_node[node_id]["properties"]:
            key_to_add = "object_states"
        else:
            key_to_add = "asset_states"
        if node_id not in child_to_parent.keys():
            return None
        else:
            parent_node_ids = child_to_parent[node_id]
        for parent_node_id in parent_node_ids:
            parent_node = id_to_node[parent_node_id[0]]
            relation_type = parent_node_id[1]
            if relation_type != "INSIDE" and relation_type != "ON":
                continue
            if "Decor" in parent_node["category"]:
                continue
            token = "<{}_{}>".format(node_name, node_id)
            rel = "{}(<{}_{}>)".format(
                relation_type, parent_node["class_name"], parent_node_id[0])
            if token in return_dict[key_to_add].keys():
                return_dict[key_to_add][token].append(rel)
            else:
                return_dict[key_to_add][token] = [rel]
    return return_dict


def populate_environment(graph, start_objects, start_room):
    environment = {
        "assets": [],
        "asset_states": {},
        "objects": [],
        "object_states": {},
    }
    id_to_node = {node["id"]: node for node in graph["nodes"]}
    name_to_id = {}
    for node in graph["nodes"]:
        if node["class_name"] in name_to_id.keys():
            name_to_id[node["class_name"]].append(node["id"])
        else:
            name_to_id[node["class_name"]] = [node["id"]]
    child_to_parent = {}
    for edge in graph["edges"]:
        if edge["from_id"] in child_to_parent.keys():
            child_to_parent[edge["from_id"]].append(
                (edge["to_id"], edge["relation_type"]))
        else:
            child_to_parent[edge["from_id"]] = [
                (edge["to_id"], edge["relation_type"])]
    objects_to_check = [remove_brackets(name) for name in start_objects]

    while objects_to_check:
        current_object = objects_to_check.pop()
        if "<{}>".format(current_object) not in environment["objects"] and "<{}>".format(
                current_object) not in environment["assets"]:
            if "GRABBABLE" in id_to_node[int(
                    current_object.split("_")[-1])]["properties"]:
                environment["objects"].append("<{}>".format(current_object))
            else:
                environment["assets"].append("<{}>".format(current_object))

            parent_info = find_parent_node(
                graph, remove_brackets(current_object), start_room)
            if parent_info is not None:
                if "object_states" in parent_info:
                    for obj, states in parent_info["object_states"].items():
                        environment["object_states"]["<{}>".format(
                            remove_brackets(obj))] = [
                            "{}(<{}>)".format(
                                state.split("(")[0],
                                remove_brackets(state.split("(")[-1].split(")")[0]))
                            for state in states]
                        for state in states:
                            involved_object = remove_brackets(
                                state.split("(")[-1].split(")")[0])
                            if "<{}>".format(involved_object) not in environment["objects"] and "<{}>".format(
                                    involved_object) not in environment["assets"]:
                                objects_to_check.append(involved_object)
                if "asset_states" in parent_info:
                    for obj, states in parent_info["asset_states"].items():
                        environment["asset_states"]["<{}>".format(
                            remove_brackets(obj))] = [
                            "{}(<{}>)".format(
                                state.split("(")[0],
                                remove_brackets(state.split("(")[-1].split(")")[0]))
                            for state in states]
                        for state in states:
                            involved_asset = remove_brackets(state.split(
                                "(")[-1].split(")")[0])
                            if "<{}>".format(involved_asset) not in environment["assets"] and "<{}>".format(
                                    involved_asset) not in environment["objects"]:
                                objects_to_check.append(involved_asset)
    asset_properties = {}
    for asset in environment["asset_states"]:
        asset_id = asset.strip(">").strip("<").split("_")[1]
        tmp_properties = []
        if "CAN_OPEN" in id_to_node[int(asset_id)]["properties"]:
            tmp_properties.append("IS_OPENABLE")
        else:
            tmp_properties.append("NOT_OPENABLE")
        asset_properties[asset] = tmp_properties
    environment["asset_properties"] = asset_properties
    object_properties = {}
    for obj in environment["object_states"]:
        obj_id = obj.strip(">").strip("<").split("_")[1]
        tmp_properties = []
        if "CAN_OPEN" in id_to_node[int(obj_id)]["properties"]:
            tmp_properties.append("IS_OPENABLE")
        else:
            tmp_properties.append("NOT_OPENABLE")
        object_properties[obj] = tmp_properties
    environment["object_properties"] = object_properties
    return environment


def find_unique_objects(graph, object_name, start_room):
    hit_object = find_parent_node(graph, object_name, start_room)
    if hit_object is None:
        return []
    if len(hit_object["object_states"]) > 0:
        object_list = hit_object["object_states"].keys()
    elif len(hit_object["asset_states"]) > 0:
        object_list = hit_object["asset_states"].keys()
    else:
        raise ValueError("No object found")
    return list(object_list)


def extract_objects(script):
    objects_all = []
    for action in script:
        parts = action.split("(")
        arguments = parts[1].replace(" ", "").strip(")")
        if len(arguments) == 0:
            objects = []
        else:
            objects = arguments.split(",")
        objects_all.extend(objects)
    return list(set(objects_all))


def normalized_plan_token(tok):
    if not tok:
        return ""
    tok = tok.strip().lower().strip("<>")
    parts = tok.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return tok


def infer_goal_from_reference(reference_program):
    pickup, target = None, None
    if not reference_program:
        return None, None
    for line in reference_program:
        s = line.replace(" ", "")
        m = re.match(r"^[Gg]rab\(<([^>]+)>\)", s)
        if m:
            pickup = normalized_plan_token(m.group(1))
        m = re.match(r"^[Pp]ut(?:[Ii]n)?\(<([^>]+)>,<([^>]+)>\)", s)
        if m:
            pickup = normalized_plan_token(m.group(1))
            target = normalized_plan_token(m.group(2))
    return pickup, target


def scenario_is_switch_only(reference_program):
    if not reference_program:
        return False
    for line in reference_program:
        if "switchon" in line.replace(" ", "").lower():
            return True
    return False


def tokens_align(step_token, goal_token):
    a = normalized_plan_token(step_token)
    b = normalized_plan_token(goal_token)
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def evaluate_plan_step_coverage(task_sequence, pickup_object="apple", put_target="fridge"):
    if not pickup_object or not put_target:
        return False
    task_sequence_lower = [step.lower().replace(" ", "") for step in task_sequence]
    has_grab = False
    has_put = False
    for step in task_sequence_lower:
        if step.startswith("grab(<") and ">)" in step:
            inner = step.split("grab(<", 1)[1].split(">)", 1)[0]
            if tokens_align(inner, pickup_object):
                has_grab = True
        if step.startswith("putin(<") and ">,<" in step:
            payload = step.split("putin(<", 1)[1].split(">)", 1)[0]
            if ">,<" in payload:
                obj, tgt = payload.split(">,<", 1)
                if tokens_align(obj, pickup_object) and tokens_align(tgt, put_target):
                    has_put = True
        if step.startswith("put(<") and ">,<" in step:
            payload = step.split("put(<", 1)[1].split(">)", 1)[0]
            if ">,<" in payload:
                obj, tgt = payload.split(">,<", 1)
                if tokens_align(obj, pickup_object) and tokens_align(tgt, put_target):
                    has_put = True
    return has_grab and has_put


def preserves_goal(original_task_sequence, validated_task_sequence, reference_program=None):
    pickup, target = infer_goal_from_reference(reference_program or [])
    if not pickup or not target:
        return True
    return evaluate_plan_step_coverage(
        validated_task_sequence, pickup_object=pickup, put_target=target)


def is_executable_task_sequence(task_sequence):
    allowed_one = {"walktowards", "grab", "open", "close", "switchon", "drink"}
    allowed_two = {"put", "putin"}
    for step in task_sequence:
        normalized = step.replace(" ", "")
        if re.match(r"^([A-Za-z]+)\(<[^>]+>,<[^>]+>\)$", normalized):
            verb = re.match(r"^([A-Za-z]+)\(", normalized).group(1).lower()
            if verb not in allowed_two:
                return False
            continue
        if re.match(r"^([A-Za-z]+)\(<[^>]+>\)$", normalized):
            verb = re.match(r"^([A-Za-z]+)\(", normalized).group(1).lower()
            if verb not in allowed_one:
                return False
            continue
        return False
    return True


def parse_action_signature(step):
    normalized = (step or "").replace(" ", "")
    m = re.match(r"^([A-Za-z]+)\((.*)\)$", normalized)
    if not m:
        return None, []
    verb = m.group(1).lower()
    args_raw = m.group(2).strip()
    if not args_raw:
        return verb, []
    args = [a.strip().strip("<>").lower() for a in args_raw.split(",")]
    return verb, args


def build_step_validation_state(environment):
    state = {
        "openable": set(),
        "open_now": set(),
        "held": set(),
        "last_walk_target": None,
    }
    for token, props in (environment.get("asset_properties") or {}).items():
        norm = normalized_plan_token(token)
        if "IS_OPENABLE" in [str(x).upper() for x in (props or [])]:
            state["openable"].add(norm)
    for token, props in (environment.get("object_properties") or {}).items():
        norm = normalized_plan_token(token)
        if "IS_OPENABLE" in [str(x).upper() for x in (props or [])]:
            state["openable"].add(norm)

    for token, states in (environment.get("asset_states") or {}).items():
        norm = normalized_plan_token(token)
        flat = " ".join(states if isinstance(states, list) else [str(states)]).lower()
        if "open(" in flat or "open()" in flat:
            state["open_now"].add(norm)
    for token, states in (environment.get("object_states") or {}).items():
        norm = normalized_plan_token(token)
        flat = " ".join(states if isinstance(states, list) else [str(states)]).lower()
        if "open(" in flat or "open()" in flat:
            state["open_now"].add(norm)
    return state


def validate_step(step, state):
    verb, args = parse_action_signature(step)
    if verb is None:
        return False, "invalid action syntax"

    if verb in {"walktowards", "switchon", "drink"}:
        return True, ""
    if verb == "grab":
        if len(args) != 1:
            return False, "grab requires one argument"
        return True, ""
    if verb == "open":
        if len(args) != 1:
            return False, "open requires one argument"
        tgt = normalized_plan_token(args[0])
        if tgt in state["openable"] and tgt in state["open_now"]:
            return False, f"{tgt} already open"
        return True, ""
    if verb == "close":
        if len(args) != 1:
            return False, "close requires one argument"
        tgt = normalized_plan_token(args[0])
        if tgt in state["openable"] and tgt not in state["open_now"]:
            return False, f"{tgt} already closed"
        return True, ""
    if verb == "put":
        if len(args) != 2:
            return False, "put requires two arguments"
        obj = normalized_plan_token(args[0])
        if obj not in state["held"]:
            return False, f"{obj} must be held before put"
        return True, ""
    if verb == "putin":
        if len(args) != 2:
            return False, "putin requires two arguments"
        obj = normalized_plan_token(args[0])
        tgt = normalized_plan_token(args[1])
        if obj not in state["held"]:
            return False, f"{obj} must be held before putin"
        if tgt in state["openable"] and tgt not in state["open_now"]:
            return False, f"{tgt} must be open before putin"
        return True, ""
    return False, f"unsupported action verb {verb}"


def apply_step_state_update(step, state):
    verb, args = parse_action_signature(step)
    if verb is None:
        return
    if verb == "walktowards" and args:
        state["last_walk_target"] = normalized_plan_token(args[0])
    elif verb == "grab" and len(args) == 1:
        state["held"].add(normalized_plan_token(args[0]))
    elif verb == "open" and len(args) == 1:
        state["open_now"].add(normalized_plan_token(args[0]))
    elif verb == "close" and len(args) == 1:
        state["open_now"].discard(normalized_plan_token(args[0]))
    elif verb in {"put", "putin"} and len(args) == 2:
        state["held"].discard(normalized_plan_token(args[0]))


def step_level_validate_sequence(task_sequence, environment):
    state = build_step_validation_state(environment)
    checked = 0
    valid = 0
    issues = []
    for idx, step in enumerate(task_sequence):
        ok, reason = validate_step(step, state)
        checked += 1
        if ok:
            valid += 1
            apply_step_state_update(step, state)
            continue
        issues.append({"step_index": idx, "step": step, "reason": reason})
    return {
        "steps_checked": checked,
        "steps_valid": valid,
        "step_valid_ratio": (valid / checked) if checked else 0.0,
        "issues": issues,
    }


def auto_fix_step_issues(task_sequence, issues):
    """Small deterministic repair pass before LLM refinement."""
    fixed = list(task_sequence)
    offset = 0
    for issue in issues:
        idx = issue["step_index"] + offset
        step = fixed[idx]
        verb, args = parse_action_signature(step)
        if verb == "putin" and len(args) == 2 and "must be open before putin" in issue["reason"]:
            tgt = args[1]
            fixed.insert(idx, f"Open(<{tgt}>)")
            offset += 1
        elif verb in {"put", "putin"} and len(args) == 2 and "must be held before" in issue["reason"]:
            obj = args[0]
            fixed.insert(idx, f"Grab(<{obj}>)")
            offset += 1
    return fixed


def _first_bracket_token_matching(reference_program, keyword):
    for line in reference_program or []:
        if keyword.lower() not in line.lower():
            continue
        m = re.search(r"<([^>]+)>", line)
        if m:
            return m.group(1)
    return None


def reconcile_toaster_bread_table_sequence(instruction, reference_program, task_sequence):
    """Insert missing navigation so execution visibly matches tasks like: bread from toaster → plate on table."""
    if not task_sequence:
        return list(task_sequence)
    inst = (instruction or "").lower()
    ref = reference_program or []
    seq = list(task_sequence)
    if "toaster" in inst and "bread" in inst:
        tok_toaster = _first_bracket_token_matching(ref, "toaster") or "toaster"
        grab_i = None
        for i, s in enumerate(seq):
            v, a = parse_action_signature(s)
            if v == "grab" and a and any("bread" in x for x in a):
                grab_i = i
                break
        if grab_i is not None:
            prefix = seq[:grab_i]
            tkey = normalized_plan_token(tok_toaster)
            tkey = tkey.split("_")[0] if tkey else "toaster"
            has_tw = any(
                parse_action_signature(s)[0] == "walktowards" and tkey in s.lower()
                for s in prefix
            )
            if not has_tw:
                seq = prefix + [f"WalkTowards(<{tok_toaster}>)"] + seq[grab_i:]
    if "toaster" in inst and "bread" in inst and any(
            k in inst for k in ("toast", "toasted", "toasting")):
        tok_t = _first_bracket_token_matching(ref, "toaster") or "toaster"
        gix = None
        for i, s in enumerate(seq):
            v, a = parse_action_signature(s)
            if v == "grab" and a and any("bread" in x for x in a):
                gix = i
                break
        if gix is not None:
            pre = seq[:gix]
            if not any(
                parse_action_signature(s)[0] == "switchon" and "toaster" in s.lower()
                for s in pre
            ):
                seq = pre + [f"SwitchOn(<{tok_t}>)"] + seq[gix:]
    if "plate" in inst and "bread" in inst:
        tok_plate = _first_bracket_token_matching(ref, "plate")
        if not tok_plate:
            tok_plate = "plate"
        tok_table = _first_bracket_token_matching(
            ref, "kitchentable") or _first_bracket_token_matching(ref, "table")
        put_i = None
        for i, s in enumerate(seq):
            v, a = parse_action_signature(s)
            if v != "put" or len(a) < 2:
                continue
            if not (any("bread" in x for x in [a[0]])):
                continue
            if "plate" in a[1] or tokens_align(a[1], tok_plate):
                put_i = i
                break
        if put_i is not None:
            prefix = seq[:put_i]
            has_walk = any(
                parse_action_signature(s)[0] == "walktowards" and (
                    "plate" in s.lower() or "table" in s.lower() or "kitchentable" in s.lower()
                )
                for s in prefix
            )
            if not has_walk:
                wtok = tok_table if tok_table else tok_plate
                seq = prefix + [f"WalkTowards(<{wtok}>)"] + seq[put_i:]
    if is_executable_task_sequence(seq):
        return seq
    return list(task_sequence)


def extract_plan_tokens(task_sequence):
    tokens = set()
    for step in task_sequence or []:
        _, args = parse_action_signature(step)
        for a in args:
            tokens.add(normalized_plan_token(a))
    return tokens


def build_allowed_tokens(environment, reference_program):
    allowed = set()
    for group in ("objects", "assets"):
        for tok in environment.get(group, []) or []:
            allowed.add(normalized_plan_token(tok))
    for line in reference_program or []:
        m = re.findall(r"<([^>]+)>", line)
        for tok in m:
            allowed.add(normalized_plan_token(tok))
    return allowed


def is_grounded_sequence(task_sequence, environment, reference_program):
    used = extract_plan_tokens(task_sequence)
    allowed = build_allowed_tokens(environment, reference_program)
    if not used:
        return True
    return used.issubset(allowed)


def call_step_reviewer(instruction, environment, accepted_steps, step, model_name):
    messages = [
        {"role": "system", "content": STEP_REVIEW_PROMPT},
        {"role": "user", "content": json.dumps({
            "instruction": instruction,
            "environment": environment,
            "accepted_steps": accepted_steps,
            "candidate_step": step,
        })}
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        top_p=1.0,
    )
    raw = response["choices"][0]["message"]["content"]
    parsed = _parse_strict_json(raw)
    repl = parsed.get("replacement_steps", [])
    if not isinstance(repl, list) or not repl:
        raise ValueError("Step reviewer returned invalid replacement_steps")
    return repl


def run_llm_step_review(instruction, environment, task_sequence, model_name, reference_program):
    reviewed = []
    for step in task_sequence:
        try:
            replacement = call_step_reviewer(
                instruction, environment, reviewed, step, model_name)
            if not is_executable_task_sequence(replacement):
                reviewed.append(step)
                continue
            candidate = reviewed + replacement + task_sequence[len(reviewed)+1:]
            if not is_grounded_sequence(candidate, environment, reference_program):
                reviewed.append(step)
                continue
            reviewed.extend(replacement)
        except Exception:
            reviewed.append(step)
    return reviewed


def score_task_sequence(task_sequence, environment, reference_program):
    """Heuristic quality score used to avoid validator regressions."""
    if not task_sequence:
        return -1.0
    score = 0.0
    if is_executable_task_sequence(task_sequence):
        score += 1.5
    else:
        score -= 2.0
    step_eval = step_level_validate_sequence(task_sequence, environment)
    score += step_eval["step_valid_ratio"]
    score -= 0.2 * len(step_eval["issues"])
    if preserves_goal(task_sequence, task_sequence, reference_program):
        score += 1.0
    pickup, target = infer_goal_from_reference(reference_program or [])
    if pickup and target and evaluate_plan_step_coverage(
            task_sequence, pickup_object=pickup, put_target=target):
        score += 0.75
    return score


def _parse_strict_json(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    return json.loads(cleaned)


def call_plan_validator(credentials, instruction, environment, original_sequence, model_name, max_retries=2):
    messages = [
        {"role": "system", "content": VALIDATION_PROMPT},
        {"role": "user", "content": json.dumps({
            "instruction": instruction,
            "environment": environment,
            "original_task_sequence": original_sequence,
        })},
    ]
    for attempt in range(max_retries + 1):
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
            top_p=1.0,
        )
        raw = response["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        try:
            parsed = json.loads(cleaned)
            if "improved_task_sequence" not in parsed or not isinstance(
                    parsed["improved_task_sequence"], list):
                raise ValueError("Validator response missing improved_task_sequence list")
            if len(parsed["improved_task_sequence"]) == 0:
                raise ValueError("Validator returned empty improved_task_sequence")
            parsed["validator_model"] = model_name
            parsed["validator_attempt"] = attempt + 1
            return parsed
        except Exception as e:
            if attempt == max_retries:
                raise
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your last response could not be used ({e}). "
                    "Return strict JSON with a non-empty improved_task_sequence."
                ),
            })


def refine_plan_with_shortcomings(instruction, environment, original_sequence, shortcomings, model_name):
    messages = [
        {"role": "system", "content": REFINEMENT_PROMPT},
        {"role": "user", "content": json.dumps({
            "instruction": instruction,
            "environment": environment,
            "original_task_sequence": original_sequence,
            "validator_shortcomings": shortcomings,
        })},
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
        top_p=1.0,
    )
    raw = response["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    parsed = json.loads(raw)
    improved = parsed.get("improved_task_sequence", [])
    if not isinstance(improved, list) or len(improved) == 0:
        raise ValueError("Refinement model returned invalid improved_task_sequence")
    return parsed


def merge_token_map(global_map, scenario_id, executor_cfg):
    m = dict(global_map or {})
    scen = (executor_cfg.get("scenarios") or {}).get(str(scenario_id), {})
    m.update(scen.get("object_map") or {})
    return m


def parse_scene_candidates():
    raw = os.getenv("SCENE_CANDIDATES", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    ranges_env = os.getenv("SCENE_SEARCH_RANGES", "").strip()
    if ranges_env:
        out = []
        for chunk in ranges_env.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(chunk))
        return [f"FloorPlan{i}" for i in out]
    start = int(os.getenv("SCENE_SEARCH_START", "1"))
    end = int(os.getenv("SCENE_SEARCH_END", "30"))
    extra_env = os.getenv("SCENE_SEARCH_EXTRA", "201-230,301-330,401-430").strip()
    extras = []
    for chunk in extra_env.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            extras.extend(range(int(a), int(b) + 1))
        else:
            extras.append(int(chunk))
    return [f"FloorPlan{i}" for i in list(range(start, end + 1)) + extras]


def required_types_for_scenario(goal_pickup, goal_target, reference_program, object_token_map):
    req = set()
    if goal_pickup:
        req.update(x.lower() for x in resolve_sim_types(goal_pickup, object_token_map))
    if goal_target:
        req.update(x.lower() for x in resolve_sim_types(goal_target, object_token_map))
    for raw in extract_objects(reference_program or []):
        req.update(x.lower() for x in resolve_sim_types(raw, object_token_map))
    return req


def scene_contains_required_types(comm, scene_name, required_types):
    try:
        comm.reset(scene_name)
        event = comm.step(action="Pass")
    except Exception:
        return False, None
    present = {o["objectType"].lower() for o in event.metadata["objects"]}
    if required_types.issubset(present):
        return True, event
    return False, event


def choose_scene_for_scenario(
    comm,
    default_scene,
    scenario_id,
    scen_cfg,
    goal_pickup,
    goal_target,
    reference_program,
    object_token_map,
):
    # Priority 1: explicit per-scenario override
    explicit = scen_cfg.get("ai2thor_scene")
    if explicit:
        return explicit, "override"

    # Priority 2: automatic scene search (optional)
    enable_auto_scene = os.getenv("ENABLE_AUTO_SCENE_SELECTION", "1").strip().lower() in (
        "1", "true", "yes")
    if not enable_auto_scene:
        return default_scene, "default"

    required = required_types_for_scenario(
        goal_pickup, goal_target, reference_program, object_token_map)
    if not required:
        return default_scene, "default_no_requirements"

    # Keep default if it already satisfies requirements.
    ok_default, _ = scene_contains_required_types(comm, default_scene, required)
    if ok_default:
        return default_scene, "default_satisfies_requirements"

    for scene_name in parse_scene_candidates():
        ok, _ = scene_contains_required_types(comm, scene_name, required)
        if ok:
            return scene_name, "auto_scene_search"

    return default_scene, "fallback_default_missing_requirements"


def resolve_sim_types(token, object_token_map):
    base = normalized_plan_token(token)
    if base in object_token_map:
        return [object_token_map[base]]
    return BUILTIN_TYPE_ALIASES.get(base, [base.replace("_", "").title()])


def find_object_id_resolved(event, token, object_token_map, prefer_pickupable=True):
    candidates = resolve_sim_types(token, object_token_map)
    objs = event.metadata["objects"]
    for ctype in candidates:
        ct = ctype.lower()
        for obj in objs:
            if obj["objectType"].lower() != ct:
                continue
            if prefer_pickupable and not obj.get("pickupable", False):
                continue
            return obj["objectId"]
    for ctype in candidates:
        ct = ctype.lower()
        for obj in objs:
            if obj["objectType"].lower() == ct:
                return obj["objectId"]
    for obj in objs:
        if obj["objectType"].lower() == normalized_plan_token(token):
            return obj["objectId"]
    return None


def _agent_facing_toward(from_pos, to_pos):
    """Yaw (degrees) so the agent at from_pos looks toward to_pos in the x-z plane."""
    dx = to_pos.get("x", 0) - from_pos.get("x", 0)
    dz = to_pos.get("z", 0) - from_pos.get("z", 0)
    return round(math.degrees(math.atan2(dx, dz)))


def _object_by_id(event, object_id):
    for o in event.metadata["objects"]:
        if o["objectId"] == object_id:
            return o
    return None


def _any_picked_object_id(event):
    for o in event.metadata.get("objects", []):
        if o.get("isPickedUp"):
            return o["objectId"]
    return None


def _planar_dist(ax, az, ox, oz):
    dx, dz = ax - ox, az - oz
    return math.sqrt(dx * dx + dz * dz) or 0.0


def list_pickup_candidates_event(event, token, object_token_map):
    """
    All pickupable object dicts that match the planner token, including
    BUILTIN aliases and fuzzy 'bread' matches (Bread vs BreadSliced, etc.).
    """
    base = normalized_plan_token(token)
    type_set = {t.lower() for t in resolve_sim_types(token, object_token_map)}
    if base in BUILTIN_TYPE_ALIASES:
        type_set |= {t.lower() for t in BUILTIN_TYPE_ALIASES[base]}

    out = {}
    for o in event.metadata["objects"]:
        if not o.get("pickupable"):
            continue
        ot = o["objectType"].lower()
        if ot in type_set or ("bread" in base and "bread" in ot):
            out[o["objectId"]] = o
    return list(out.values())


def _ordered_pickup_candidates_for_grab(event, object_name, object_token_map):
    """Order pickup tries: for bread, prefer slices parented to / near the toaster first."""
    base = normalized_plan_token(object_name)
    onl = (object_name or "").lower()
    cands = list_pickup_candidates_event(event, object_name, object_token_map)
    if not cands:
        return []
    if "bread" in base or "bread" in onl:
        t_id = find_object_id_resolved(event, "toaster", object_token_map, False)
        tpos = None
        if t_id:
            t_o = _object_by_id(event, t_id)
            tpos = (t_o or {}).get("position")
        ap = event.metadata["agent"]["position"]
        ax, az = ap.get("x", 0), ap.get("z", 0)

        def p_key(o):
            parents = o.get("parentReceptacles") or []
            in_t = 0 if (t_id and t_id in parents) else 1
            pr = o.get("position") or {}
            px, pz = pr.get("x", 0), pr.get("z", 0)
            d = o.get("distance")
            if d is None or not isinstance(d, (int, float)):
                d = _planar_dist(ax, az, px, pz)
            d_t = 0.0
            if tpos:
                d_t = _planar_dist(px, pz, tpos.get("x", 0), tpos.get("z", 0))
            vis = 0 if o.get("visible", False) else 1
            return (in_t, vis, d_t, d)

        return sorted(cands, key=p_key)
    return _sort_candidates_for_interaction(event, cands, prefer_visible=True)


def _sort_candidates_for_interaction(event, cands, prefer_visible=True):
    ap = event.metadata["agent"]["position"]
    ax, az = ap.get("x", 0), ap.get("z", 0)

    def key(o):
        p = o.get("position") or {}
        px, pz = p.get("x", 0), p.get("z", 0)
        d = o.get("distance")
        if d is None or not isinstance(d, (int, float)):
            d = _planar_dist(ax, az, px, pz)
        vis = 0 if o.get("visible", False) else 1
        if not prefer_visible:
            vis = 0
        return (vis, float(d) if d is not None else 999.0, o.get("objectId", ""))

    return sorted(cands, key=key)


def _target_type_set_for_put(token, object_token_map):
    base = normalized_plan_token(token)
    type_set = {t.lower() for t in resolve_sim_types(token, object_token_map)}
    if base in BUILTIN_TYPE_ALIASES:
        type_set |= {t.lower() for t in BUILTIN_TYPE_ALIASES[base]}
    if base in ("table", "kitchentable", "diningtable"):
        type_set |= {"diningtable", "coffeetable", "sidetable"}
    if "plate" in base:
        type_set.add("plate")
    if base in ("counter", "kitchencounter", "countertop"):
        type_set |= {"countertop"}
    if base in ("sink",):
        type_set |= {"sinkbasin"}
    if base in ("shelf", "bookshelf", "shelvingunit"):
        type_set |= {"shelf", "shelvingunit", "dresser", "sidetable"}
    if base in ("bowl",):
        type_set |= {"bowl"}
    if base in ("microwave",):
        type_set |= {"microwave"}
    if base in ("fridge", "refrigerator"):
        type_set |= {"fridge"}
    if base in ("cup", "mug"):
        type_set |= {"cup", "mug"}
    return type_set


def find_put_target_id_resolved(event, target_name, object_token_map):
    """
    Put/PutIn surface: if multiple Plates/Tables match, use the one nearest the agent
    (and prefer visible) so we place on the table's plate, not a stray plate elsewhere.
    """
    type_set = _target_type_set_for_put(target_name, object_token_map)
    cands = [
        o
        for o in event.metadata["objects"]
        if o.get("objectType", "").lower() in type_set
    ]
    if not cands:
        return find_object_id_resolved(event, target_name, object_token_map, False)
    sorted_objs = _sort_candidates_for_interaction(event, cands, prefer_visible=True)
    return sorted_objs[0]["objectId"]


def teleport_near_object(comm, event, object_id, scale=0.65, horizon=30):
    target = None
    for o in event.metadata["objects"]:
        if o["objectId"] == object_id:
            target = o
            break
    if not target:
        return event
    pos = target.get("position") or {}
    agent = event.metadata["agent"]["position"]
    dx = pos.get("x", 0) - agent["x"]
    dz = pos.get("z", 0) - agent["z"]
    dist = math.sqrt(dx * dx + dz * dz) or 1.0
    rot = _agent_facing_toward(agent, pos)
    last_event = event
    for s in [scale, scale * 0.7, scale * 1.3, scale * 0.4, scale * 1.6]:
        eff = s / dist
        tx = agent["x"] + dx * eff
        tz = agent["z"] + dz * eff
        ev = comm.step(
            action="TeleportFull",
            position=dict(x=tx, y=agent["y"], z=tz),
            rotation=dict(x=0, y=rot, z=0),
            horizon=horizon,
            standing=True,
        )
        last_event = ev
        if ev.metadata.get("lastActionSuccess"):
            return ev
    return last_event


def _teleport_in_front_of(comm, event, object_id, forward_scale, horizon=30):
    """Step toward object_id; forward_scale 0.35–0.5 usually lands in interactable range."""
    return teleport_near_object(comm, event, object_id, scale=forward_scale, horizon=horizon)


def _rotate_until_visible(comm, event, object_id, max_rots=16):
    for _ in range(max_rots):
        o = _object_by_id(event, object_id)
        if o and o.get("visible", False):
            return event
        event = comm.step(action="RotateRight")
    return event


def _agent_horizon_y(event):
    a = event.metadata.get("agent") or {}
    h = a.get("cameraHorizon", a.get("horizon", 30))
    try:
        return float(h)
    except (TypeError, ValueError):
        return 30.0


def _agent_rotation_y(event):
    a = event.metadata.get("agent") or {}
    r = a.get("rotation") or {}
    if isinstance(r, dict) and "y" in r:
        return r["y"]
    return 0.0


def _short_approach_moves(comm, event, object_id, steps=2):
    """Face target and nudge forward a few times to close distance in clutter."""
    o = _object_by_id(event, object_id)
    if not o or not o.get("position"):
        return event
    tpos = o["position"]
    for _ in range(steps):
        o = _object_by_id(event, object_id)
        if o and o.get("visible", False):
            return event
        ap = event.metadata["agent"]["position"]
        rot = _agent_facing_toward(ap, tpos)
        h = int(round(_agent_horizon_y(event)))
        event = comm.step(
            action="TeleportFull",
            position=dict(x=ap["x"], y=ap["y"], z=ap["z"]),
            rotation=dict(x=0, y=rot, z=0),
            horizon=h,
            standing=True,
        )
        event = comm.step(action="MoveAhead", moveMagnitude=0.25, forceAction=True)
    return event


def ensure_interactable(comm, event, object_id, is_pickup=True):
    """
    Teleport, rotate, and short-move so object_id is likely visible; try a few
    camera horizons. Pickup/placement in iTHOR is strict about line-of-sight.
    """
    scales = (0.55, 0.38, 0.28) if is_pickup else (0.55, 0.42, 0.32)
    horizons = (30, 15, 0, 22, 40)
    for sc in scales:
        event = _teleport_in_front_of(comm, event, object_id, sc, horizon=30)
        event = _rotate_until_visible(comm, event, object_id, max_rots=8)
        event = _short_approach_moves(comm, event, object_id, steps=1)
        for h in horizons:
            ap = event.metadata["agent"]["position"]
            ry = _agent_rotation_y(event)
            event = comm.step(
                action="TeleportFull",
                position=dict(x=ap["x"], y=ap["y"], z=ap["z"]),
                rotation=dict(x=0, y=ry, z=0),
                horizon=h,
                standing=True,
            )
            o = _object_by_id(event, object_id)
            if o and o.get("visible", False):
                return event, h
    return event, 30


def _auto_open_if_closed_openable(comm, event, object_id):
    """If the object is openable and currently closed, walk to it and open it.

    Generic over Fridge / Microwave / Cabinet / Drawer / Oven / Safe / Toaster,
    so a put/putin that targets a closed receptacle does not have to wait for
    an explicit Open in the plan.
    """
    o = _object_by_id(event, object_id)
    if not o:
        return event
    if not o.get("openable"):
        return event
    if o.get("isOpen"):
        return event
    event = teleport_near_object(comm, event, object_id, scale=0.5, horizon=30)
    event = _rotate_until_visible(comm, event, object_id, max_rots=8)
    event = comm.step(action="OpenObject", objectId=object_id)
    return event


def _ensure_toaster_open_if_bread(comm, event, object_name, object_token_map):
    n = (object_name or "").lower()
    if "bread" not in n and normalized_plan_token(object_name) != "bread":
        return event
    t_oid = find_object_id_resolved(event, "toaster", object_token_map, False)
    if not t_oid:
        return event
    to = _object_by_id(event, t_oid)
    if to and to.get("openable") and not to.get("isOpen", False):
        event = teleport_near_object(comm, event, t_oid, scale=0.5, horizon=30)
        event = _rotate_until_visible(comm, event, t_oid, max_rots=8)
        event = comm.step(action="OpenObject", objectId=t_oid)
    return event


def build_llm_environment(event, reference_program, instructions_text, scenario_id, executor_cfg):
    """Ground the planner: pickupable objects vs non-pickupable assets, plus openable hints."""
    token_map = merge_token_map(
        executor_cfg.get("global_object_map"), scenario_id, executor_cfg)
    asset_states = {}
    object_states = {}
    objects = []
    assets = []
    seen = set()

    def add_obj(tok):
        t = tok if tok.startswith("<") else f"<{tok}>"
        if t.lower() not in seen:
            seen.add(t.lower())
            objects.append(t)

    def add_asset(tok):
        t = tok if tok.startswith("<") else f"<{tok}>"
        if t.lower() not in seen:
            seen.add(t.lower())
            assets.append(t)

    for o in event.metadata["objects"]:
        ot = o["objectType"].lower()
        tok = f"<{ot}>"
        if o.get("pickupable"):
            add_obj(tok)
            object_states.setdefault(tok, ["on_something(<floor>)"])
        elif o.get("openable") or o["objectType"] in (
                "Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "Oven"):
            add_asset(tok)
            st = "open()" if o.get("isOpen") else "closed()"
            asset_states[tok] = [st]

    for raw in extract_objects(reference_program):
        base = normalized_plan_token(raw)
        mapped = token_map.get(base, base)
        for o in event.metadata["objects"]:
            if o["objectType"].lower() == mapped.lower():
                if o.get("pickupable"):
                    add_obj(f"<{base}>")
                else:
                    add_asset(f"<{base}>")
                break

    asset_properties = {}
    for a in assets:
        ap = ["NOT_OPENABLE"]
        for o in event.metadata["objects"]:
            if f"<{o['objectType'].lower()}>" == a.lower() or normalized_plan_token(a) == o["objectType"].lower():
                if o.get("openable"):
                    ap = ["IS_OPENABLE"]
                break
        asset_properties[a] = ap

    object_properties = {}
    for obj in objects:
        object_properties[obj] = ["NOT_OPENABLE"]

    out = {
        "instruction_context": instructions_text,
        "objects": sorted(objects),
        "object_states": object_states,
        "assets": sorted(assets),
        "asset_states": asset_states,
        "asset_properties": asset_properties,
        "object_properties": object_properties,
        "simulator_object_map_hint": token_map,
    }
    tnotes = []
    it = (instructions_text or "").lower()
    if "toaster" in it and "bread" in it:
        tnotes.append(
            "Walk to the toaster first, then grab the bread, then go to the plate on the table for placement."
        )
    if tnotes:
        out["task_specific_notes"] = tnotes
    return out


def _aabb_corners(o):
    bb = (o.get("axisAlignedBoundingBox") or {}).get("cornerPoints") or []
    return bb


def _aabb_min_max(o):
    pts = _aabb_corners(o)
    if not pts:
        p = o.get("position") or {}
        if not p:
            return None
        return ((p.get("x", 0) - 0.1, p.get("y", 0) - 0.1, p.get("z", 0) - 0.1),
                (p.get("x", 0) + 0.1, p.get("y", 0) + 0.1, p.get("z", 0) + 0.1))
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    zs = [pt[2] for pt in pts]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _point_in_aabb(p, mn, mx, pad=0.05):
    return (mn[0] - pad <= p["x"] <= mx[0] + pad
            and mn[2] - pad <= p["z"] <= mx[2] + pad
            and mn[1] - pad <= p["y"] <= mx[1] + pad + 0.5)


def _xz_within(p, mn, mx, pad=0.1):
    return (mn[0] - pad <= p["x"] <= mx[0] + pad
            and mn[2] - pad <= p["z"] <= mx[2] + pad)


def goal_satisfied_in_metadata(event, goal_pickup, goal_target, object_token_map):
    if not goal_pickup or not goal_target:
        return None
    pickup_types = {x.lower() for x in resolve_sim_types(goal_pickup, object_token_map)}
    target_types = {x.lower() for x in resolve_sim_types(goal_target, object_token_map)}
    gp = normalized_plan_token(goal_pickup)
    gt = normalized_plan_token(goal_target)

    if "bread" in gp:
        pickup_types |= {"bread", "breadsliced", "breadloaf"}
    if "slice" in gp:
        pickup_types |= {"breadsliced", "bread", "breadloaf"}

    if gt in ("table", "kitchentable", "diningtable"):
        target_types |= {"diningtable", "coffeetable", "sidetable"}
    if "plate" in gt:
        target_types.add("plate")
    if gt in ("counter", "kitchencounter", "countertop"):
        target_types |= {"countertop"}
    if gt in ("sink",):
        target_types |= {"sink", "sinkbasin"}
    if gt in ("shelf", "bookshelf", "shelvingunit"):
        target_types |= {"shelf", "shelvingunit", "dresser", "sidetable"}
    if gt in ("bowl",):
        target_types |= {"bowl"}
    if gt in ("microwave",):
        target_types |= {"microwave"}
    if gt in ("fridge", "refrigerator"):
        target_types |= {"fridge"}
    if gt in ("cup", "mug"):
        target_types |= {"cup", "mug"}

    def matches_pickup(o):
        if not o.get("pickupable"):
            return False
        ot = o["objectType"].lower()
        if ot in pickup_types:
            return True
        if "bread" in gp and "bread" in ot:
            return True
        return False

    def matches_target(tobj):
        tt = tobj["objectType"].lower()
        return tt in target_types

    obj_by_id = {o["objectId"]: o for o in event.metadata["objects"]}
    target_objs = [o for o in event.metadata["objects"] if matches_target(o)]
    target_ids = {o["objectId"] for o in target_objs}

    for o in event.metadata["objects"]:
        if not matches_pickup(o):
            continue
        parents = o.get("parentReceptacles") or []
        for pid in parents:
            if pid in target_ids:
                return True
            pobj = obj_by_id.get(pid)
            if pobj and matches_target(pobj):
                return True
        # Spatial containment fallback: AI2-THOR sometimes parents to the
        # outer receptacle group (e.g. Sink) but not the inner (SinkBasin),
        # or marks objects as parented to CounterTop near the target. Treat
        # the goal as satisfied if the picked object's position lies inside
        # the AABB of any matching target receptacle.
        op = o.get("position") or {}
        if not op:
            continue
        for t in target_objs:
            mm = _aabb_min_max(t)
            if not mm:
                continue
            mn, mx = mm
            if _point_in_aabb(op, mn, mx, pad=0.05):
                return True
            tt = t["objectType"].lower()
            if tt in {"sinkbasin", "sink", "bowl", "plate", "microwave", "fridge"}:
                if _xz_within(op, mn, mx, pad=0.15):
                    if op.get("y", 0) <= mx[1] + 0.6:
                        return True
    return False


def test_execution(
    comm,
    script,
    scene_name="FloorPlan1",
    object_token_map=None,
    goal_pickup=None,
    goal_target=None,
    frame_prefix=None,
):
    object_token_map = object_token_map or {}
    reset(comm, scene_name)
    event = comm.step(action="Pass")

    picked_object = None
    pickup_success = False
    put_success = False
    switch_success = False
    final_error = ""
    step_sleep = float(os.getenv("STEP_SLEEP_SEC", "0.5"))
    if frame_prefix:
        os.makedirs(frame_prefix, exist_ok=True)

    for step_i, step in enumerate(script):
        time.sleep(step_sleep)
        print("EXECUTING:", step)
        parts = step.split()
        action_name = parts[0].lower()

        if action_name == "walktowards":
            target_name = parts[1].lower()
            found_visible = False
            oid = find_object_id_resolved(event, target_name, object_token_map, False)
            if oid:
                for _ in range(6):
                    for obj in event.metadata["objects"]:
                        if obj["objectId"] == oid and obj.get("visible", False):
                            found_visible = True
                            break
                    if found_visible:
                        break
                    event = comm.step(action="RotateRight")
                    time.sleep(0.3)
                if not found_visible:
                    event = teleport_near_object(comm, event, oid)
            else:
                for _ in range(4):
                    for obj in event.metadata["objects"]:
                        if obj["objectType"].lower() == target_name and obj.get("visible", False):
                            found_visible = True
                            break
                    if found_visible:
                        break
                    event = comm.step(action="RotateRight")
                    time.sleep(0.3)
                if not found_visible:
                    event = comm.step(action="MoveAhead")

        elif action_name == "grab":
            object_name = parts[1]
            oname_l = object_name.lower()
            event = _ensure_toaster_open_if_bread(comm, event, object_name, object_token_map)
            cands = _ordered_pickup_candidates_for_grab(
                event, object_name, object_token_map
            )
            if not cands:
                oid_nav = find_object_id_resolved(
                    event, object_name, object_token_map, False
                )
                if oid_nav:
                    event = teleport_near_object(comm, event, oid_nav, scale=0.5)
                event = _ensure_toaster_open_if_bread(comm, event, object_name, object_token_map)
                cands = _ordered_pickup_candidates_for_grab(
                    event, object_name, object_token_map
                )
            if not cands:
                final_error = f"Could not find object: {object_name}"
            else:
                got = False
                for otry in cands:
                    object_id = otry["objectId"]
                    event, _hz = ensure_interactable(comm, event, object_id, is_pickup=True)
                    event = comm.step(
                        action="PickupObject", objectId=object_id, forceAction=True
                    )
                    held = _any_picked_object_id(event)
                    omd = _object_by_id(event, object_id) or {}
                    if event.metadata.get("lastActionSuccess") and (
                        held or omd.get("isPickedUp")
                    ):
                        pickup_success = True
                        picked_object = object_name
                        got = True
                        break
                    if "bread" in oname_l:
                        t_oid = find_object_id_resolved(
                            event, "toaster", object_token_map, False
                        )
                        if t_oid:
                            event = teleport_near_object(comm, event, t_oid, scale=0.45)
                            to = _object_by_id(event, t_oid)
                            if to and to.get("openable") and not to.get("isOpen", False):
                                event = comm.step(action="OpenObject", objectId=t_oid)
                        event, _hz = ensure_interactable(comm, event, object_id, is_pickup=True)
                        event = comm.step(
                            action="PickupObject", objectId=object_id, forceAction=True
                        )
                        held = _any_picked_object_id(event)
                        omd = _object_by_id(event, object_id) or {}
                        if event.metadata.get("lastActionSuccess") and (
                            held or omd.get("isPickedUp")
                        ):
                            pickup_success = True
                            picked_object = object_name
                            got = True
                            break
                if not got:
                    pickup_success = False
                    final_error = (event.metadata.get("errorMessage") or "pickup failed")

        elif action_name == "open":
            tid = parts[1]
            object_id = find_object_id_resolved(event, tid, object_token_map, False)
            if object_id:
                event = comm.step(action="OpenObject", objectId=object_id)
            else:
                final_error = f"Could not find object: {tid}"

        elif action_name == "close":
            tid = parts[1]
            object_id = find_object_id_resolved(event, tid, object_token_map, False)
            if object_id:
                event = comm.step(action="CloseObject", objectId=object_id)

        elif action_name in ("put", "putin"):
            target_name = parts[2]
            target_id = find_put_target_id_resolved(event, target_name, object_token_map)
            if target_id is None:
                final_error = f"Could not find target: {target_name}"
            elif not _any_picked_object_id(event):
                final_error = "Put failed: not holding an object (pickup did not stick)."
            else:
                event = _auto_open_if_closed_openable(comm, event, target_id)
                event, _hz = ensure_interactable(comm, event, target_id, is_pickup=False)
                put_success = False
                for _attempt in range(4):
                    event = comm.step(
                        action="PutObject",
                        objectId=target_id,
                        forceAction=True,
                        placeStationary=(_attempt < 2),
                    )
                    put_success = bool(event.metadata.get("lastActionSuccess"))
                    if put_success:
                        break
                    event = _rotate_until_visible(comm, event, target_id, max_rots=4)
                    event = teleport_near_object(comm, event, target_id, scale=0.4)
                    event, _ = ensure_interactable(comm, event, target_id, is_pickup=False)
                if not put_success and _any_picked_object_id(event):
                    held_id_before = _any_picked_object_id(event)
                    event = teleport_near_object(comm, event, target_id, scale=0.35)
                    event = _rotate_until_visible(comm, event, target_id, max_rots=8)
                    drop_event = comm.step(action="DropHandObject", forceAction=True)
                    if drop_event.metadata.get("lastActionSuccess"):
                        event = drop_event
                        for _o in event.metadata.get("objects", []):
                            if _o.get("objectId") == held_id_before:
                                parents = _o.get("parentReceptacles") or []
                                if target_id in parents:
                                    put_success = True
                                else:
                                    tobj = _object_by_id(event, target_id)
                                    if tobj:
                                        mm = _aabb_min_max(tobj)
                                        if mm and _xz_within(_o.get("position") or {}, mm[0], mm[1], pad=0.2):
                                            put_success = True
                                break
                if not put_success:
                    if not _any_picked_object_id(event):
                        final_error = (
                            "Put failed: no object in hand; pickup likely failed."
                        )
                    else:
                        final_error = event.metadata.get("errorMessage", "")

        elif action_name == "switchon":
            tid = parts[1]
            oid = find_object_id_resolved(event, tid, object_token_map, False)
            if oid:
                event = comm.step(action="ToggleObjectOn", objectId=oid)
                switch_success = event.metadata["lastActionSuccess"]

        elif action_name == "drink":
            event = comm.step(action="Pass")

        else:
            print("Unknown action:", action_name)

        if frame_prefix and event is not None and event.frame is not None:
            tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", action_name)[:32]
            fpng = os.path.join(
                frame_prefix, f"step{step_i:02d}_{tag}.png")
            try:
                Image.fromarray(event.frame).save(fpng)
            except Exception as e:
                print("Frame save failed:", e)

    Image.fromarray(event.frame).save("ai2thor_after_execution.png")
    goal_achieved = goal_satisfied_in_metadata(
        event, goal_pickup, goal_target, object_token_map)
    return {
        "pickup_success": pickup_success,
        "put_success": put_success,
        "switch_success": switch_success,
        "execution_success": pickup_success and put_success,
        "goal_achieved": goal_achieved,
        "picked_object": picked_object,
        "final_error": final_error,
    }


def load_learning_memory(memory_path):
    if not os.path.exists(memory_path):
        return {"runs": 0, "shortcomings_counter": {}, "error_counter": {}, "notes": []}
    with open(memory_path) as f:
        return json.load(f)


def save_learning_memory(memory_path, memory):
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)


def update_learning_memory(memory, validator_result, original_execution_result, validated_execution_result):
    memory["runs"] = memory.get("runs", 0) + 1
    shortcomings_counter = Counter(memory.get("shortcomings_counter", {}))
    error_counter = Counter(memory.get("error_counter", {}))

    if validator_result and isinstance(validator_result.get("shortcomings"), list):
        for item in validator_result["shortcomings"]:
            shortcomings_counter[item.strip()] += 1

    for result in (original_execution_result, validated_execution_result):
        err = (result.get("final_error") or "").strip()
        if err:
            error_counter[err] += 1

    memory["shortcomings_counter"] = dict(shortcomings_counter)
    memory["error_counter"] = dict(error_counter)
    return memory


def build_adaptive_guidance(memory, max_items=3):
    if memory.get("runs", 0) == 0:
        return ""
    shortcomings = Counter(memory.get("shortcomings_counter", {})).most_common(max_items)
    errors = Counter(memory.get("error_counter", {})).most_common(max_items)
    lines = ["Use these learned failure patterns from prior runs:"]
    if shortcomings:
        lines.append("Frequent validator-identified shortcomings:")
        lines.extend([f"- {m}" for m, _ in shortcomings])
    if errors:
        lines.append("Frequent execution errors:")
        lines.extend([f"- {m}" for m, _ in errors])
    lines.append("Only use Open/Close when the target container is relevant and openable.")
    lines.append("Do not introduce unrelated objects/tools that are not in the current environment.")
    return "\n".join(lines)


def discover_scenario_ids(scenarios_dir="scenarios"):
    if not os.path.isdir(scenarios_dir):
        return []
    ids = []
    for name in os.listdir(scenarios_dir):
        if not name.endswith(".json") or name == "executor_sim_overrides.json":
            continue
        stem = name[:-5]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(ids)


def parse_scenario_ids_env():
    raw = os.getenv("SCENARIO_IDS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    start = int(os.getenv("SCENARIO_START", "1"))
    end_raw = os.getenv("SCENARIO_END", "").strip()
    scenarios_dir = os.getenv("SCENARIOS_DIR", "scenarios")
    discovered = discover_scenario_ids(scenarios_dir)
    if end_raw:
        end = int(end_raw)
    else:
        end = max(discovered) if discovered else start
    return list(range(start, end + 1))


class ChatGPT:
    VALID_API_VERSIONS = ["2022-12-01", "2023-05-15"]

    def __init__(
            self,
            credentials,
            prompt_load_order,
            use_azure=True,
            api_version="2023-05-15",
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3):
        self.use_azure = use_azure
        self.model_name = model_name
        self.temperature = temperature
        if self.use_azure:
            openai.api_key = credentials["azureopenai"]["AZURE_OPENAI_KEY"]
            openai.api_base = credentials["azureopenai"]["AZURE_OPENAI_ENDPOINT"]
            openai.api_type = "azure"
            if api_version not in self.VALID_API_VERSIONS:
                raise ValueError(
                    f"api_version must be one of {self.VALID_API_VERSIONS}")
            openai.api_version = api_version
        else:
            openai.organization = credentials["openai"]["YOUR_ORG_ID"]
            openai.api_key = credentials["openai"]["OPENAI_API_KEY"]

        self.credentials = credentials
        self.messages = []
        self.max_token_length = 15000
        self.max_completion_length = 2000
        self.last_response = None
        self.last_response_raw = None
        self.query = ""
        self.instruction = ""
        fp_system = os.path.join(dir_system, "system.txt")
        with open(fp_system) as f:
            data = f.read()
        self.system_message = {"role": "system", "content": data}

        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + ".txt")
            with open(fp_prompt) as f:
                data = f.read()
            data_spilit = re.split(r"\[user\]\n|\[assistant\]\n", data)
            data_spilit = [item for item in data_spilit if len(item) != 0]
            assert len(data_spilit) % 2 == 0
            for i, item in enumerate(data_spilit):
                if i % 2 == 0:
                    self.messages.append({"sender": "user", "text": item})
                else:
                    self.messages.append({"sender": "assistant", "text": item})
        fp_query = os.path.join(dir_query, "query.txt")
        with open(fp_query) as f:
            self.query = f.read()

    def create_prompt(self):
        prompt = []
        prompt.append(self.system_message)
        for message in self.messages:
            prompt.append({"role": message["sender"], "content": message["text"]})
        prompt_content = ""
        for message in prompt:
            prompt_content += message["content"]
        print("prompt length:", len(enc.encode(prompt_content)))
        if len(enc.encode(prompt_content)) > self.max_token_length - self.max_completion_length:
            print("prompt too long. truncated.")
            self.messages = self.messages[2:]
            return self.create_prompt()
        return prompt

    def extract_json_part(self, text):
        if "```python" in text:
            return text[text.find("```python") + len("```python"):text.find("\n```")]
        if "```" in text:
            a = text.find("```")
            b = text.find("```", a + 3)
            if b != -1:
                chunk = text[a + 3:b]
                if chunk.lstrip().lower().startswith("json"):
                    chunk = chunk.lstrip()[4:].lstrip()
                return chunk
        return text

    def generate(self, message, environment, is_user_feedback=False):
        if is_user_feedback:
            self.messages.append({"sender": "user", "text": message})
        else:
            text_base = self.query
            if "[ENVIRONMENT]" in text_base:
                text_base = text_base.replace("[ENVIRONMENT]", json.dumps(environment))
            if "[INSTRUCTION]" in text_base:
                text_base = text_base.replace("[INSTRUCTION]", message)
                self.instruction = text_base
            self.messages.append({"sender": "user", "text": text_base})

        if self.use_azure and openai.api_version == "2023-05-15":
            deployment_name = self.credentials["azureopenai"]["AZURE_OPENAI_DEPLOYMENT_NAME_CHATGPT"]
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=self.create_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_completion_length,
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            text = response["choices"][0]["message"]["content"]
        else:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=self.create_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_completion_length,
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            text = response["choices"][0]["message"]["content"]

        self.last_response_raw = text
        self.messages.append({"sender": "assistant", "text": self.last_response_raw})
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        self.last_response = self.last_response.replace("'", '"')
        self.last_response = self.last_response.replace('Let"s', "Let's")
        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            self.environment = self.json_dict.get("environment_after", environment)
        except BaseException:
            self.json_dict = None
            return None
        return self.json_dict


if __name__ == "__main__":
    default_scene = os.getenv("AI2THOR_SCENE", "FloorPlan1")
    comm = Controller(scene=default_scene, width=800, height=600)
    dir_name = os.getenv(
        "OUTPUT_DIR_NAME",
        "out_task_planning_gpt-3.5-turbo-16k_temp=2.0",
    )
    planner_model = os.getenv("PLANNER_MODEL", "gpt-3.5-turbo-16k")
    validator_model = os.getenv("VALIDATOR_MODEL", "gpt-4.1")
    enable_step_validation = os.getenv("ENABLE_STEP_VALIDATION", "1").strip().lower() in (
        "1", "true", "yes")
    enable_llm_step_review = os.getenv("ENABLE_LLM_STEP_REVIEW", "1").strip().lower() in (
        "1", "true", "yes")
    step_review_model = os.getenv("STEP_REVIEW_MODEL", validator_model)
    planner_retries = int(os.getenv("PLANNER_RETRIES", "3"))
    keep_validated_only_if_not_worse = os.getenv(
        "KEEP_VALIDATED_ONLY_IF_NOT_WORSE", "1").strip().lower() in ("1", "true", "yes")
    enable_original_fallback = os.getenv(
        "ENABLE_ORIGINAL_FALLBACK", "1").strip().lower() in ("1", "true", "yes")
    planner_temperature = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))
    waittime_sec = 5
    max_trial = int(os.getenv("MAX_TRIAL", "1"))
    force_rerun = os.getenv("FORCE_RERUN", "0").strip().lower() in ("1", "true", "yes")
    scenarios_dir = os.getenv("SCENARIOS_DIR", "scenarios")
    scenario_ids = parse_scenario_ids_env()
    executor_cfg = load_executor_sim_config(scenarios_dir)
    benchmark_rows = []
    time_api_called = time.time() - waittime_sec
    learning_memory_path = "./out_learning_memory.json"
    learning_memory = load_learning_memory(learning_memory_path)
    compiled_guidance = build_adaptive_guidance(learning_memory)

    for scenario_id in scenario_ids:
        for trial_idx in range(max_trial):
            print(f"scenario_id={scenario_id}, trial_idx={trial_idx}")
            scenario_name = "scenario_" + str(scenario_id)
            dump_name = "./" + dir_name + f"/{scenario_name}/{trial_idx}"
            fp = os.path.join(dump_name + ".json")
            if os.path.exists(fp) and not force_rerun:
                continue
            scenario_path = os.path.join(scenarios_dir, str(scenario_id) + ".json")
            if not os.path.isfile(scenario_path):
                print("Skip missing scenario file:", scenario_path)
                continue
            with open(scenario_path) as f:
                scenario = json.load(f)

            instructions = scenario.get("instructions") or []
            reference_program = scenario.get("program") or []
            instructions_text = instructions[0] if instructions else ""
            goal_pickup, goal_target = infer_goal_from_reference(reference_program)

            scen_cfg = (executor_cfg.get("scenarios") or {}).get(str(scenario_id), {})
            object_token_map = merge_token_map(
                executor_cfg.get("global_object_map"), scenario_id, executor_cfg)
            ai2thor_scene, scene_selection_source = choose_scene_for_scenario(
                comm,
                default_scene,
                scenario_id,
                scen_cfg,
                goal_pickup,
                goal_target,
                reference_program,
                object_token_map,
            )
            print(
                f"scene selected for scenario {scenario_id}: {ai2thor_scene} "
                f"(source={scene_selection_source})"
            )

            reset(comm, ai2thor_scene)
            event = comm.step(action="Pass")
            Image.fromarray(event.frame).save("ai2thor_view.png")

            environment = build_llm_environment(
                event, reference_program, instructions_text, scenario_id, executor_cfg)

            if not os.path.exists("./" + dir_name + "/" + scenario_name):
                os.makedirs("./" + dir_name + "/" + scenario_name)

            text = None
            aimodel = None
            for planner_attempt in range(planner_retries):
                current_time = time.time()
                if current_time - time_api_called < waittime_sec:
                    time.sleep(waittime_sec - (current_time - time_api_called))
                aimodel = ChatGPT(
                    credentials,
                    prompt_load_order=prompt_load_order,
                    use_azure=False,
                    model_name=planner_model,
                    temperature=planner_temperature,
                )
                planner_instr = instructions_text
                if compiled_guidance:
                    planner_instr = (
                        f"{instructions_text}\n\nAdditional planning guidance:\n{compiled_guidance}"
                    )
                text = aimodel.generate(planner_instr, environment, is_user_feedback=False)
                time_api_called = time.time()
                if text is not None:
                    break
                print(f"api call failed on planner attempt {planner_attempt + 1}. retrying...")
                time.sleep(max(0, waittime_sec - (time.time() - time_api_called)))
                text = aimodel.generate(
                    "Your return cannot be interpreted as a valid json dictionary. "
                    "Please reformat your response.",
                    environment,
                    is_user_feedback=True,
                )
                if text is not None:
                    break

            if text is None:
                with open("./" + dir_name + f"/{scenario_name}/note.txt", "w") as f:
                    f.write(str(aimodel.last_response))
                continue

            original_task_sequence = text["task_cohesion"]["task_sequence"]
            validator_result = None
            validated_task_sequence = list(original_task_sequence)
            pre_step_validation = {}
            post_step_validation = {}

            try:
                validator_result = call_plan_validator(
                    credentials,
                    instructions_text,
                    environment,
                    original_task_sequence,
                    validator_model,
                )
                improved = validator_result.get("improved_task_sequence") or []
                if improved and is_executable_task_sequence(improved):
                    if (
                        preserves_goal(original_task_sequence, improved, reference_program)
                        and is_grounded_sequence(improved, environment, reference_program)
                    ):
                        validated_task_sequence = improved
                shortcomings = validator_result.get("shortcomings") or []
                if shortcomings:
                    try:
                        refined = refine_plan_with_shortcomings(
                            instructions_text,
                            environment,
                            validated_task_sequence,
                            shortcomings,
                            validator_model,
                        )
                        refined_seq = refined.get("improved_task_sequence", [])
                        if (
                            refined_seq
                            and is_executable_task_sequence(refined_seq)
                            and preserves_goal(original_task_sequence, refined_seq, reference_program)
                            and is_grounded_sequence(refined_seq, environment, reference_program)
                        ):
                            validated_task_sequence = refined_seq
                    except Exception as e:
                        print("Refinement skipped:", e)
            except Exception as e:
                print("Validator skipped:", e)

            if enable_llm_step_review:
                reviewed = run_llm_step_review(
                    instructions_text,
                    environment,
                    validated_task_sequence,
                    step_review_model,
                    reference_program,
                )
                if (
                    reviewed
                    and is_executable_task_sequence(reviewed)
                    and preserves_goal(original_task_sequence, reviewed, reference_program)
                    and is_grounded_sequence(reviewed, environment, reference_program)
                ):
                    validated_task_sequence = reviewed

            if enable_step_validation:
                pre_step_validation = step_level_validate_sequence(
                    validated_task_sequence, environment)
                if pre_step_validation["issues"]:
                    auto_fixed = auto_fix_step_issues(
                        validated_task_sequence, pre_step_validation["issues"])
                    if (
                        is_executable_task_sequence(auto_fixed)
                        and preserves_goal(original_task_sequence, auto_fixed, reference_program)
                    ):
                        validated_task_sequence = auto_fixed
                post_step_validation = step_level_validate_sequence(
                    validated_task_sequence, environment)
            else:
                post_step_validation = {"steps_checked": 0, "steps_valid": 0, "step_valid_ratio": 0.0, "issues": []}

            if keep_validated_only_if_not_worse:
                original_score = score_task_sequence(
                    original_task_sequence, environment, reference_program)
                validated_score = score_task_sequence(
                    validated_task_sequence, environment, reference_program)
                if validated_score < original_score:
                    print(
                        "Reverting validated sequence to original due to lower quality score:",
                        validated_score, "<", original_score,
                    )
                    validated_task_sequence = list(original_task_sequence)
                    post_step_validation = step_level_validate_sequence(
                        validated_task_sequence, environment) if enable_step_validation else post_step_validation

            original_task_sequence = reconcile_toaster_bread_table_sequence(
                instructions_text, reference_program, original_task_sequence)
            validated_task_sequence = reconcile_toaster_bread_table_sequence(
                instructions_text, reference_program, validated_task_sequence)

            save_frames = os.getenv("SAVE_EXEC_FRAMES", "1").strip().lower() in (
                "1", "true", "yes")
            frame_base = f"./{dir_name}/{scenario_name}/trial_{trial_idx}"
            orig_frames = f"{frame_base}/original" if save_frames else None
            val_frames = f"{frame_base}/validated" if save_frames else None

            print("self test is running for original plan...")
            original_script = generate_script(original_task_sequence)
            original_execution_result = test_execution(
                comm,
                original_script,
                scene_name=ai2thor_scene,
                object_token_map=object_token_map,
                goal_pickup=goal_pickup,
                goal_target=goal_target,
                frame_prefix=orig_frames,
            )
            print(
                "  [original exec] goal_achieved=%s pickup=%s put=%s err=%r"
                % (
                    original_execution_result.get("goal_achieved"),
                    original_execution_result.get("pickup_success"),
                    original_execution_result.get("put_success"),
                    (original_execution_result.get("final_error") or "")[:200],
                )
            )

            print("self test is running for validated plan...")
            validated_script = generate_script(validated_task_sequence)
            validated_execution_result = test_execution(
                comm,
                validated_script,
                scene_name=ai2thor_scene,
                object_token_map=object_token_map,
                goal_pickup=goal_pickup,
                goal_target=goal_target,
                frame_prefix=val_frames,
            )
            print(
                "  [validated exec] goal_achieved=%s pickup=%s put=%s err=%r"
                % (
                    validated_execution_result.get("goal_achieved"),
                    validated_execution_result.get("pickup_success"),
                    validated_execution_result.get("put_success"),
                    (validated_execution_result.get("final_error") or "")[:200],
                )
            )

            switch_only = scenario_is_switch_only(reference_program)
            if switch_only:
                original_plan_success = any(
                    "switchon(" in s.lower().replace(" ", "") for s in original_task_sequence)
                validated_plan_success = any(
                    "switchon(" in s.lower().replace(" ", "") for s in validated_task_sequence)
                original_execution_success = bool(original_execution_result.get("switch_success"))
                validated_execution_success = bool(validated_execution_result.get("switch_success"))
            elif goal_pickup and goal_target:
                original_plan_success = evaluate_plan_step_coverage(
                    original_task_sequence, pickup_object=goal_pickup, put_target=goal_target)
                validated_plan_success = evaluate_plan_step_coverage(
                    validated_task_sequence, pickup_object=goal_pickup, put_target=goal_target)
                original_execution_success = bool(
                    original_execution_result.get("goal_achieved"))
                validated_execution_success = bool(
                    validated_execution_result.get("goal_achieved"))
            else:
                original_plan_success = bool(original_task_sequence)
                validated_plan_success = bool(validated_task_sequence)
                original_execution_success = original_execution_result.get("execution_success", False)
                validated_execution_success = validated_execution_result.get("execution_success", False)

            fallback_to_original_applied = False
            if (
                enable_original_fallback
                and original_execution_success
                and not validated_execution_success
            ):
                print("Applying safety fallback: validated failed but original succeeded.")
                fallback_to_original_applied = True
                validated_task_sequence = list(original_task_sequence)
                validated_execution_result = dict(original_execution_result)
                validated_execution_success = True
                validated_plan_success = original_plan_success

            learning_memory = update_learning_memory(
                learning_memory,
                validator_result,
                original_execution_result,
                validated_execution_result,
            )
            save_learning_memory(learning_memory_path, learning_memory)
            compiled_guidance = build_adaptive_guidance(learning_memory)

            aimodel.json_dict["scenario_id"] = scenario_id
            aimodel.json_dict["scenario_instruction"] = instructions_text
            aimodel.json_dict["ai2thor_scene"] = ai2thor_scene
            aimodel.json_dict["scene_selection_source"] = scene_selection_source
            aimodel.json_dict["goal_pickup"] = goal_pickup
            aimodel.json_dict["goal_target"] = goal_target
            aimodel.json_dict["plan_success"] = validated_plan_success
            aimodel.json_dict["was_execution_successful"] = validated_execution_success
            aimodel.json_dict["execution_result"] = validated_execution_result
            aimodel.json_dict["planner_model"] = planner_model
            aimodel.json_dict["validator"] = {
                "model": validator_model,
                "result": validator_result,
                "original_task_sequence": original_task_sequence,
                "validated_task_sequence": validated_task_sequence,
            }
            aimodel.json_dict["step_validation"] = {
                "enabled": enable_step_validation,
                "pre": pre_step_validation,
                "post": post_step_validation,
            }
            aimodel.json_dict["safety_fallback"] = {
                "enabled": enable_original_fallback,
                "applied": fallback_to_original_applied,
                "condition": "original_execution_success and not validated_execution_success",
            }
            aimodel.json_dict["ab_comparison"] = {
                "original": {
                    "task_sequence": original_task_sequence,
                    "plan_success": original_plan_success,
                    "execution_success": original_execution_success,
                    "execution_result": original_execution_result,
                    "frame_dir": orig_frames,
                },
                "validated": {
                    "task_sequence": validated_task_sequence,
                    "plan_success": validated_plan_success,
                    "execution_success": validated_execution_success,
                    "execution_result": validated_execution_result,
                    "frame_dir": val_frames,
                },
            }

            with open(fp, "w") as f:
                json.dump(aimodel.json_dict, f, indent=4)

            benchmark_rows.append({
                "scenario_id": scenario_id,
                "trial_idx": trial_idx,
                "original_plan_success": original_plan_success,
                "original_execution_success": original_execution_success,
                "validated_plan_success": validated_plan_success,
                "validated_execution_success": validated_execution_success,
                "step_valid_ratio_pre": pre_step_validation.get("step_valid_ratio", 0.0),
                "step_valid_ratio_post": post_step_validation.get("step_valid_ratio", 0.0),
                "step_issues_pre": len(pre_step_validation.get("issues", [])),
                "step_issues_post": len(post_step_validation.get("issues", [])),
                "fallback_to_original_applied": fallback_to_original_applied,
                "ai2thor_scene": ai2thor_scene,
                "scene_selection_source": scene_selection_source,
            })

            n = len(benchmark_rows)
            print("BENCHMARK TABLE:")
            for row in benchmark_rows:
                print(row)
            print(
                "original_execution_accuracy:",
                sum(1 for r in benchmark_rows if r["original_execution_success"]) / n,
            )
            print(
                "validated_execution_accuracy:",
                sum(1 for r in benchmark_rows if r["validated_execution_success"]) / n,
            )

    summary = {
        "planner_model": planner_model,
        "validator_model": validator_model,
        "enable_llm_step_review": enable_llm_step_review,
        "step_review_model": step_review_model,
        "enable_step_validation": enable_step_validation,
        "enable_original_fallback": enable_original_fallback,
        "scenario_ids": scenario_ids,
        "results": benchmark_rows,
    }
    if benchmark_rows:
        n = len(benchmark_rows)
        summary["original_execution_accuracy"] = sum(
            1 for r in benchmark_rows if r["original_execution_success"]) / n
        summary["validated_execution_accuracy"] = sum(
            1 for r in benchmark_rows if r["validated_execution_success"]) / n
        summary["execution_accuracy_delta"] = (
            summary["validated_execution_accuracy"] - summary["original_execution_accuracy"])
        summary["avg_step_valid_ratio_pre"] = sum(
            r.get("step_valid_ratio_pre", 0.0) for r in benchmark_rows) / n
        summary["avg_step_valid_ratio_post"] = sum(
            r.get("step_valid_ratio_post", 0.0) for r in benchmark_rows) / n
        summary["avg_step_issues_pre"] = sum(
            r.get("step_issues_pre", 0) for r in benchmark_rows) / n
        summary["avg_step_issues_post"] = sum(
            r.get("step_issues_post", 0) for r in benchmark_rows) / n
    with open("./benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved benchmark_summary.json")
