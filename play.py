import torch
import numpy as np
from game.game import QuartoGame
from agent.dqn_agent import DQNAgent
from game.piece import Piece

# Action Conversion (copied from train.py for self-containment)
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

def print_board(board):
    """Prints the Quarto board to the console."""
    print("  0 1 2 3")
    for i, row in enumerate(board.board):
        print(f"{i} ", end="")
        for piece in row:
            if piece is None:
                print(". ", end="")
            else:
                print(f"{piece.piece_id:x} ", end="") # Print piece ID in hex
        print()

def get_player_input(prompt, valid_options=None):
    """
    Gets and validates player input.
    valid_options can be a range or a set.
    """
    while True:
        try:
            user_input = input(prompt).strip()
            if user_input.lower() == 'q': # Allow quitting
                return 'q'
            value = int(user_input, 16) if "(in hex)" in prompt else int(user_input)
            if valid_options is None or value in valid_options:
                return value
            else:
                print(f"Invalid input. Please enter a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number (or 'q' to quit).")

def get_piece_attributes(piece_id):
    """Returns a human-readable string of a piece's attributes."""
    if piece_id is None:
        return ""
    # Attributes from piece.py: Color, Height, Shape, Hollowness
    attributes = format(piece_id, '04b')
    color = "Dark" if attributes[0] == '1' else "Light"
    height = "Tall" if attributes[1] == '1' else "Short"
    shape = "Square" if attributes[2] == '1' else "Round"
    hollowness = "Hollow" if attributes[3] == '1' else "Solid"
    return f"({color}, {height}, {shape}, {hollowness})"

def main(model_path):
    """
    Main function to run the human vs. AI game.
    """
    state_size = 19 # 4x4 board + piece_to_place + current_player + game_phase
    action_size = 32 # 16 placement cells + 16 pieces to select
    
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load(model_path)
        agent.epsilon = 0
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return

    game = QuartoGame()
    human_player = 0

    print("--- Welcome to Quarto! ---")
    
    # First move: Human selects a piece for the AI.
    print_board(game.board)
    print(f"\n--- Your Turn (Player {human_player}) ---")
    print("You start by selecting a piece for the AI to place.")
    available_pieces_hex = sorted([f"{p:x}" for p in game.board.available_pieces])
    print("Available pieces:", available_pieces_hex)
    piece_id = get_player_input("Select a piece for the AI to place (in hex, or 'q' to quit): ", game.board.available_pieces)
    if piece_id == 'q': return
    game.select_piece(piece_id)
    print(f"You selected piece {piece_id:x} {get_piece_attributes(piece_id)} for AI.")
    
    game.current_player = 1 - human_player

    # Main game loop
    while not game.is_game_over():
        is_human_turn = game.current_player == human_player

        if is_human_turn:
            print(f"\n--- Your Turn (Player {human_player}) ---")
            
            # 1. Place piece selected by AI
            print_board(game.board)
            print(f"You must place piece {game.piece_to_place:x} {get_piece_attributes(game.piece_to_place)}.")
            while True:
                row = get_player_input("Enter row (0-3, or 'q' to quit): ", range(4))
                if row == 'q': return
                col = get_player_input("Enter column (0-3, or 'q' to quit): ", range(4))
                if col == 'q': return
                
                if game.board.is_valid_placement(game.piece_to_place, row, col):
                    break
                else:
                    print("Invalid placement. That cell is already occupied.")
            
            piece_id_placed = game.piece_to_place
            game.place_piece(row, col)
            print(f"You placed piece {piece_id_placed:x} at ({row}, {col}).")

            if game.is_game_over(): break

            # 2. Select piece for AI
            print_board(game.board)
            available_pieces_hex = sorted([f"{p:x}" for p in game.board.available_pieces])
            print("Available pieces:", available_pieces_hex)
            piece_id_selected = get_player_input("Select a piece for the AI to place (in hex, or 'q' to quit): ", game.board.available_pieces)
            if piece_id_selected == 'q': return
            game.select_piece(piece_id_selected)
            print(f"You selected piece {piece_id_selected:x} {get_piece_attributes(piece_id_selected)} for AI.")

        else: # AI's turn
            ai_player = 1 - human_player
            print(f"\n--- AI's Turn (Player {ai_player}) ---")

            # 1. Place piece selected by Human
            print_board(game.board)
            print(f"AI must place piece {game.piece_to_place:x} {get_piece_attributes(game.piece_to_place)}.")
            valid_placement_actions = [get_index_from_action('place', (r, c)) for r in range(4) for c in range(4) if game.board.is_valid_placement(game.piece_to_place, r, c)]
            
            if not valid_placement_actions:
                print("AI has no valid placement moves. It's a draw.")
                game.game_over = True
                break 

            action_idx = agent.act(game.get_state(), valid_placement_actions)
            _, (row, col) = get_action_from_index(action_idx)
            
            piece_id_placed = game.piece_to_place
            game.place_piece(row, col)
            print(f"AI placed piece {piece_id_placed:x} at ({row}, {col}).")

            if game.is_game_over(): break

            # 2. Select piece for Human
            print_board(game.board)
            print("AI is selecting a piece for you...")
            valid_selection_actions = [get_index_from_action('select', piece_id) for piece_id in game.board.available_pieces]

            if not valid_selection_actions:
                print("No more pieces to select. It's a draw.")
                game.game_over = True
                break
            
            state_for_selection = game.get_state()
            action_idx = agent.act(state_for_selection, valid_selection_actions)
            _, piece_id_selected = get_action_from_index(action_idx)
            
            game.select_piece(piece_id_selected)
            print(f"AI selected piece {piece_id_selected:x} {get_piece_attributes(piece_id_selected)} for you.")

        game.switch_player()

    print("\n--- GAME OVER ---")
    print_board(game.board)
    if game.winner is not None:
        winner_name = "Human" if game.winner == human_player else "AI"
        print(f"Congratulations! {winner_name} wins!")
    elif game.board.is_full() or not game.board.available_pieces:
        print("It's a draw!")
    else:
        print("Game finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play Quarto against a trained DQN agent.")
    parser.add_argument("model_path", help="Path to the trained model file (.pth).")
    args = parser.parse_args()
    main(args.model_path)