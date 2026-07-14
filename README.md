# ChatGPT-Robot-Manipulation-Prompts (Extended)

Independent MIT-licensed extension of Microsoft’s
[ChatGPT-Robot-Manipulation-Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts)
with stronger planning constraints and an AI2-THOR / VirtualHome-style evaluation loop.
**Not affiliated with or endorsed by Microsoft.**

The upstream project provides prompts that can be used with OpenAI's ChatGPT to enable natural language communication between humans and robots for executing tasks. The prompts are designed to allow ChatGPT to convert natural language instructions into a sequence of executable robot actions, with a focus on robot manipulation tasks. The prompts are easy to customize and integrate with existing robot control and visual recognition systems.
For more information, please see the original [blog post](https://www.microsoft.com/en-us/research/group/applied-robotics-research/articles/gpt-models-meet-robotic-applications-long-step-robot-control-in-various-environments/) and paper, [ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application](https://ieeexplore.ieee.org/document/10235949).


![overview](./img/overview.jpg)
## How to use
> 🚀 **New Feature Alert**: We've updated the prompts to support the OpenAI's official API. Additionally, we've updated the prompts to support the latest version of the Azure OpenAI's API (as of September 2023).
1. We provide sample codes for using ChatGPT through [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview) and [OpenAI API](https://platform.openai.com/docs/api-reference). Copy [`secrets.example.json`](./secrets.example.json) to `secrets.json` and fill in your credentials (never commit `secrets.json`). Even if you do not have a subscription, you can try it out by copying and pasting the prompts into the [OpenAI's interface](https://chat.openai.com/).

2. If you have a subscription of Azure OpenAI or OpenAI, install the required python packages by running the following command in a terminal session (note: we have confirmed that the sample codes work with python 3.9.16):
```bash
> pip install -r requirements.txt
```
Then, go to a subfolder in [examples/](./examples) (for example, [examples/task_decomposition](./examples/task_decomposition)), run the following command to run the sample code:
```bash
python aimodel.py --scenario <scenario_name>
```
Replace `<scenario_name>` with the name of the scenario you want to run. Specific scenario names can be found in the `aimodel.py` argument help.

### Extensions (planning quality and evaluation)

The original [ChatGPT-Robot-Manipulation-Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts) project focuses on prompt-driven decomposition into the high-level robot action vocabulary. In practice, plans often fail when mapped to simulators or hardware because receptacles are treated as already open, object names drift from the environment dict, or step lists fall out of sync with actions.

This codebase extends that idea as follows:

1. **Author prompts (`examples/task_decomposition`)**  
   - Added [`prompt_planning_constraints.txt`](./examples/task_decomposition/prompt/prompt_planning_constraints.txt) to the default prompt load order so the same planner receives explicit feasibility rules (open/close receptacles, name hygiene, linear executable steps, goal preservation).  
   - Corrected Example 2 in `prompt_example.txt` so `object_name` matches the instruction (`<spam>` not `<sponge>`).

2. **Embodied benchmark (`examples/task_decomposition_virtualhome`)**  
   - `task_planning.py` runs multiple scenarios, scores **original vs validated** plans, and can execute in **AI2-THOR** with object/scene overrides.  
   - A **stronger model** critiques plans (shortcomings + improved sequence), optional **refinement** and **execution repair** loops, and **learning memory** plus **compiled planner rules** (`out_learning_memory.json`, `planner_learned_rules.json`) feed failures back into later planner prompts instead of hand-written validator code.

Run from `examples/task_decomposition_virtualhome` (after `pip install -r requirements.txt` and configuring `secrets.json`):

```bash
# Example: scenarios 1–14, optional env vars for models and rerun
FORCE_RERUN=1 SCENARIO_START=1 SCENARIO_END=14 python task_planning.py
```

Useful environment variables include `PLANNER_MODEL`, `VALIDATOR_MODEL`, `RULE_COMPILER_MODEL`, `REPAIR_MODEL`, `ENABLE_RULE_COMPILER`, `ENABLE_EXECUTION_REPAIR`, `SCENARIO_IDS`, and `SCENARIOS_DIR`. Outputs include per-run JSON under your output directory and `benchmark_summary.json`.

## Bibliography
```
@article{10235949,
  author={Wake, Naoki and Kanehira, Atsushi and Sasabuchi, Kazuhiro and Takamatsu, Jun and Ikeuchi, Katsushi},
  journal={IEEE Access}, 
  title={ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2023.3310935}}
@article{wake2023gpt,
  title={GPT-4V (ision) for Robotics: Multimodal Task Planning from Human Demonstration},
  author={Wake, Naoki and Kanehira, Atsushi and Sasabuchi, Kazuhiro and Takamatsu, Jun and Ikeuchi, Katsushi},
  journal={arXiv preprint arXiv:2311.12015},
  year={2023}
}
```

## Attribution and license

This repository is a **modified derivative** of Microsoft’s open-source project
[ChatGPT-Robot-Manipulation-Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts),
originally released under the [MIT License](./LICENSE).

- Original copyright: © Microsoft Corporation
- Modifications / extensions in this fork: © 2026 Aniket Garg
- This project is **not** an official Microsoft product and is **not** affiliated with or endorsed by Microsoft.

Please keep the MIT license text and copyright notices when you redistribute.

### Secrets and credentials

Copy the template and fill in your own keys locally — do **not** commit real credentials:

```bash
cp secrets.example.json secrets.json
```

`secrets.json` is gitignored.

## Contributing

Issues and pull requests improving planning quality, evaluation, or docs are welcome.
By contributing, you agree your changes may be redistributed under the MIT License.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
