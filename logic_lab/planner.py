from collections import deque

class Action:
    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = set(pos_pre)
        self.neg_pre = set(neg_pre)
        self.pos_eff = set(pos_eff)
        self.neg_eff = set(neg_eff)

    def is_applicable(self, state):
        # Action is applicable if all positive preconditions are in the state
        # and no negative preconditions are in the state.
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        # 1. remove its negative effects from the state
        # 2. add its positive effects to the state
        new_state = set(state)
        new_state.difference_update(self.neg_eff)
        new_state.update(self.pos_eff)
        return frozenset(new_state)

def bfs_planner(initial_state, goal_state, actions):
    # Queue stores tuples of (current_state, plan_so_far, states_reached)
    queue = deque([(frozenset(initial_state), [], [frozenset(initial_state)])])
    visited = set([frozenset(initial_state)])

    while queue:
        current_state, plan, states_reached = queue.popleft()

        if goal_state.issubset(current_state):
            print("Goal reached!")
            print("Plan:")
            for step in plan:
                print(f" - {step}")
            print("\nStates reached after each action:")
            for idx, s in enumerate(states_reached):
                print(f" S{idx}: {set(s)}")
            return plan

        for action in actions:
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action.name], states_reached + [new_state]))

    print("No plan found")
    return None

if __name__ == "__main__":
    # Test A: Solvable Problem
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}

    actions = [
            Action("Move(A, B)", ["At(Robot, A)"], [], ["At(Robot, B)"], ["At(Robot, A)"]),
            Action("Move(B, A)", ["At(Robot, B)"], [], ["At(Robot, A)"], ["At(Robot, B)"]),
            Action("Move(B, C)", ["At(Robot, B)"], [], ["At(Robot, C)"], ["At(Robot, B)"]),
            Action("Move(C, B)", ["At(Robot, C)"], [], ["At(Robot, B)"], ["At(Robot, C)"]),
            Action("PickUp(Package, A)", ["At(Robot, A)", "At(Package, A)"], [], ["Holding(Package)"], ["At(Package, A)"]),
            Action("PickUp(Package, B)", ["At(Robot, B)", "At(Package, B)"], [], ["Holding(Package)"], ["At(Package, B)"]),
            Action("PickUp(Package, C)", ["At(Robot, C)", "At(Package, C)"], [], ["Holding(Package)"], ["At(Package, C)"]),
            Action("Drop(Package, A)", ["At(Robot, A)", "Holding(Package)"], [], ["At(Package, A)"], ["Holding(Package)"]),
            Action("Drop(Package, B)", ["At(Robot, B)", "Holding(Package)"], [], ["At(Package, B)"], ["Holding(Package)"]),
            Action("Drop(Package, C)", ["At(Robot, C)", "Holding(Package)"], [], ["At(Package, C)"], ["Holding(Package)"])
            ]

    print("--- Test A: Solvable Problem ---")
    bfs_planner(initial_state, goal, actions)

    # Test B: Impossible Problem (PickUp actions removed)
    print("\n--- Test B: Impossible Problem ---")
    impossible_actions = [a for a in actions if not a.name.startswith("PickUp")]
    bfs_planner(initial_state, goal, impossible_actions)

    # Test C: Irrelevant Actions
    print("\n--- Test C: Irrelevant Actions ---")
    irrelevant_actions = actions + [
            Action("Irrelevant_Move(A, B)", ["At(Robot, A)"], [], ["At(Robot, B)"], ["At(Robot, A)"])
            ]
    bfs_planner(initial_state, goal, irrelevant_actions)
