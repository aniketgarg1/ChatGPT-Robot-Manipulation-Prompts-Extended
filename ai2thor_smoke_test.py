from ai2thor.controller import Controller

controller = Controller(scene="FloorPlan1")

event = controller.step(action="MoveAhead")

print("AI2-THOR started successfully")
print("Agent position:", event.metadata["agent"]["position"])
print("Last action success:", event.metadata["lastActionSuccess"])

controller.stop()