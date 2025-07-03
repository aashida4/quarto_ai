import numpy as np
import logging
from collections import deque

from game.game import QuartoGame
from agent.dqn_agent import DQNAgent

# --- Hyperparameters ---
EPISODES = 5000
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# --- Action Conversion ---
def get_action_from_index(index):
    """Converts an action index (0-31) to a (type, value) tuple."""
    if 0 <= index < 16:
        return "select", index
    elif 16 <= index < 32:
        return "place", ((index - 16) // 4, (index - 16) % 4)
    else:
        raise ValueError("Invalid action index")

def get_index_from_action(action_type, value):
    """Converts an action (type, value) to a single index."""
    if action_type == 'select':
        return value
    elif action_type == 'place':
        row, col = value
        return row * 4 + col + 16

# --- Main Training Loop ---
def main():
    try:
        # --- Initialization ---
        logging.basicConfig(filename='selfplay_training.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
        print("Training started. Logging to selfplay_training.log")
        logging.info("Self-Play Training Started")

        game = QuartoGame()
        state_size = 19  # 4x4 board + piece_to_place + current_player + game_phase
        action_size = 32 # 16 placement cells + 16 pieces to select
        
        agent = DQNAgent(state_size, action_size)
        agent.memory = deque(maxlen=MEMORY_SIZE) # Centralized replay buffer

        # --- Self-Play Training ---
        for e in range(EPISODES):
            game.reset()
            logging.info(f"--- Starting Episode {e+1}/{EPISODES} ---")
            
            # Player 0 selects the first piece for Player 1 to place
            initial_piece = np.random.choice(list(game.board.available_pieces))
            game.select_piece(initial_piece) # This will switch current_player to 1
            logging.info(f"Game starts. Player 0 selected piece {initial_piece:x} for Player 1 to place.")

            done = False
            turn = 0
            episode_experiences = [] # Store experiences for this episode
            
            while not done:
                turn += 1
                current_player_id = game.current_player # This will be 1 for the first turn
                piece_to_place = game.piece_to_place
                logging.info(f"Turn {turn}: Player {current_player_id} to place piece {piece_to_place:x}.")
                state = game.get_state()
                logging.debug(f"[DEBUG] State for placement: {state}")

                # --- Action Phase (Placement) ---
                piece_to_place = game.piece_to_place
                
                # Get valid placement actions
                valid_placement_actions = []
                for r in range(4):
                    for c in range(4):
                        if game.board.is_valid_placement(piece_to_place, r, c):
                            valid_placement_actions.append(get_index_from_action('place', (r, c)))
                logging.debug(f"[DEBUG] Valid placement actions: {valid_placement_actions}")

                action_idx = agent.act(state, valid_placement_actions)
                action_type, action_val = get_action_from_index(action_idx)
                logging.debug(f"[DEBUG] Agent chose placement action: {action_type} {action_val}")

                # This check should now ideally always be True due to action masking
                is_valid_placement_action = False
                if action_type == 'place':
                    row, col = action_val
                    if game.board.is_valid_placement(piece_to_place, row, col):
                        is_valid_placement_action = True
                
                if not is_valid_placement_action:
                    # This block should rarely be hit if action masking works correctly
                    logging.error(f"CRITICAL ERROR: Agent chose an invalid placement action despite masking. Action: {action_type} {action_val}")
                    # Fallback to random valid placement
                    random_cell = np.argwhere(game.board.board == None)
                    row, col = random_cell[np.random.choice(len(random_cell))]
                    action_idx = get_index_from_action('place', (row, col))
                
                logging.info(f"Turn {turn}: Player {current_player_id} to place piece {game.piece_to_place:x} at ({row}, {col}).")
                game.place_piece(row, col)
                logging.info(f"Player {current_player_id} successfully placed piece {piece_to_place:x} at ({row}, {col}).")

                # Store placement experience (reward will be determined at end of episode)
                episode_experiences.append((state, action_idx, game.get_state(), game.is_game_over(), current_player_id, "place"))

                done = game.is_game_over()
                if done: break

                # --- Action Phase (Selection) ---
                state_for_selection = game.get_state()
                logging.debug(f"[DEBUG] State for selection: {state_for_selection}")
                
                # Get valid selection actions
                valid_selection_actions = []
                for piece_id in game.board.available_pieces:
                    valid_selection_actions.append(get_index_from_action('select', piece_id))
                logging.debug(f"[DEBUG] Valid selection actions: {valid_selection_actions}")

                action_idx_select = agent.act(state_for_selection, valid_selection_actions)
                action_type_select, piece_to_select = get_action_from_index(action_idx_select)
                logging.debug(f"[DEBUG] Agent chose selection action: {action_type_select} {piece_to_select}")

                # This check should now ideally always be True due to action masking
                is_valid_selection_action = False
                if action_type_select == 'select' and piece_to_select in game.board.available_pieces:
                    is_valid_selection_action = True

                if not is_valid_selection_action:
                    # This block should rarely be hit if action masking works correctly
                    logging.error(f"CRITICAL ERROR: Agent chose an invalid selection action despite masking. Action: {action_type_select} {piece_to_select}")
                    # Fallback to random valid selection
                    piece_to_select = np.random.choice(list(game.board.available_pieces))
                    action_idx_select = get_index_from_action('select', piece_to_select)
                
                game.select_piece(piece_to_select)
                logging.info(f"Player {current_player_id} selects piece {piece_to_select:x} for Player {1 - current_player_id}.")

                # Store selection experience (reward will be determined at end of episode)
                episode_experiences.append((state_for_selection, action_idx_select, game.get_state(), game.is_game_over(), current_player_id, "select"))

                # --- Learning Step ---
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
                    logging.debug("Agent replay performed.")
                
                # Switch player for the next turn
                game.switch_player()

            # --- Episode End: Re-evaluate Rewards and Add to Agent Memory ---
            final_winner = game.winner
            final_draw = game.is_game_over() and game.winner is None

            for exp_state, exp_action_idx, exp_next_state, exp_done, exp_player_id, exp_action_type in episode_experiences:
                reward = 0
                if final_winner is not None:
                    if exp_action_type == "place":
                        # If the player who placed the piece won, reward +1
                        if exp_player_id == final_winner:
                            reward = 1
                        # If the player who placed the piece lost, reward -1
                        else:
                            reward = -1
                    elif exp_action_type == "select":
                        # If the player who selected the piece caused the opponent to win, reward -1
                        # (i.e., the selector's opponent won, so the selector lost)
                        if exp_player_id != final_winner:
                            reward = 1 # The selector's opponent lost, so the selector won
                        else:
                            reward = -1 # The selector's opponent won, so the selector lost
                elif final_draw:
                    reward = 0.5 # Draw
                
                agent.remember(exp_state, exp_action_idx, reward, exp_next_state, True) # Mark as done

            logging.info(f"--- Episode {e+1} finished in {turn} turns. Winner: {final_winner if final_winner is not None else 'Draw'}. Epsilon: {agent.epsilon:.4f} ---")

            # --- Save Model ---
            if (e + 1) % 100 == 0:
                save_path = f"quarto-dqn-selfplay-episode-{e+1}.pth"
                agent.save(save_path)
                logging.info(f"Model saved to {save_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()