import numpy as np


def calculate_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])  # Manhattan distance


def extract_agents_positions_and_level(env):
    # Extract agents' positions from observations
    agents_info = [(player.position, player.level) for player in env.players]
    return agents_info


def extract_foods_positions(env):
    # Extract foods' positions from the environment
    food_positions = np.argwhere(env.field > 0)

    # Getting the level of food at each position
    food_levels = [env.field[position[0], position[1]] for position in food_positions]

    # Converting the positions to tuples
    food_positions = [tuple(position) for position in food_positions]

    food_info = []
    for position, level in zip(food_positions, food_levels):
        food_info.append((position, level))
    return food_info


def closest_pickable_food(agent_position, agent_level, foods_positions_and_levels):
    # Sort the food based on distance and pickability
    sorted_foods = sorted(
        ((pos, level) for pos, level in foods_positions_and_levels if level <= agent_level),
        key=lambda food: calculate_distance(agent_position, food[0])
    )

    return sorted_foods[0] if sorted_foods else None


def detect_pairs_heading_for_food(agents_positions_and_levels, foods_positions_and_levels):
    pairs = []
    agent_distances_to_foods = []

    # Calculate each agent's distance to each food item and store it with agent index
    for agent_idx, (agent_position, agent_level) in enumerate(agents_positions_and_levels):
        for food_position, food_level in foods_positions_and_levels:
            dist = calculate_distance(agent_position, food_position)
            agent_distances_to_foods.append((dist, agent_idx, agent_position, agent_level, food_position, food_level))

    # Sort based on distance
    agent_distances_to_foods.sort()

    selected_agents = set()
    selected_foods = set()

    for dist, agent_idx, agent_position, agent_level, food_position, food_level in agent_distances_to_foods:
        # Skip if agent or food is already selected
        if agent_idx in selected_agents or food_position in selected_foods:
            continue

        # Find the closest agent to the same food who is not the same agent
        for _, other_agent_idx, _, other_agent_level, _, _ in agent_distances_to_foods:
            if other_agent_idx != agent_idx and other_agent_idx not in selected_agents:
                summed_level = agent_level + other_agent_level
                if agent_level < food_level <= summed_level and other_agent_level < food_level:
                    # Form a pair and select this food
                    pairs.append(((agent_idx, other_agent_idx), food_position))
                    selected_agents.update([agent_idx, other_agent_idx])
                    selected_foods.add(food_position)
                    break

    return pairs


def is_pickable(agent_levels, food_level):
    agents_level = sum(agent_levels)
    return agents_level >= food_level


def find_closest_pickable_fruit(agent_position, agent_level, foods_positions_and_levels):
    closest_fruit = None
    min_distance = float('inf')

    for food_position, food_level in foods_positions_and_levels:
        distance = calculate_distance(agent_position, food_position)
        if distance < min_distance:
            min_distance = distance
            if agent_level >= food_level:
                closest_fruit = food_position

    return closest_fruit


def are_moving_towards_food(agents, food, position_history):
    # Check if there are previous actions for comparison
    if len(position_history) < 2:
        return False  # Not enough data to determine if moving towards food

    # Get the positions of the pair at the current and previous steps
    current_positions = [position_history[-1][i] for i in agents]
    previous_positions = [position_history[-2][i] for i in agents]

    all_heading_towards = True
    for i in range(len(current_positions)):
        food_distance = calculate_distance(current_positions[i], food)
        if food_distance < calculate_distance(previous_positions[i], food) or food_distance <= 1:
            continue
        else:
            all_heading_towards = False
            break

    return all_heading_towards


def is_moving_closer(agent_idx, target_pos, position_history):
    if len(position_history) < 2:
        return False  # Not enough data to determine if moving towards food

    current_position = position_history[-1][agent_idx]
    previous_position = position_history[-2][agent_idx]

    previous_distance = calculate_distance(previous_position, target_pos)
    current_distance = calculate_distance(current_position, target_pos)

    return current_distance < previous_distance
