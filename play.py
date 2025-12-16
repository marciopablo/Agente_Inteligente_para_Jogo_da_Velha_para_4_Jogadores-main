import numpy as np
import time
import os
from environment import TicTacToeEnv
from agent import QAgent
from settings import *

def print_board(board):
    """
    Função auxiliar para desenhar o tabuleiro bonito no terminal.
    Substitui os números 0, 1, 2... pelos símbolos ., X, O...
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"--- Jokenpo 4 Players AI ---")
    print(f"Agente: {SYMBOLS[AGENT_ID]} | Oponentes: {SYMBOLS[2]}, {SYMBOLS[3]}, {SYMBOLS[4]}")
    print("-" * 25)

    for i in range(BOARD_SIZE):
        row_str = " | ".join([SYMBOLS[cell] for cell in board[i]])
        print(f" {row_str} ")
        if i < BOARD_SIZE - 1:
            print("-" * (BOARD_SIZE * 4 - 1))
    print("-" * 25)

def play_demonstration():
    env = TicTacToeEnv()
    agent = QAgent()
    
    if os.path.exists("brain.pkl"):
        agent.load_model("brain.pkl")
    else:
        print("❌ Erro: brain.pkl não encontrado. Rode o train.py primeiro!")
        return

    agent.epsilon = 0.0 
    
    for game in range(1, 6):
        state = env.reset()
        state_matrix = env.board
        done = False
        print(f"\n📢 Iniciando Partida {game}...")
        time.sleep(1)
        
        step_count = 0
        while not done:
            print_board(env.board)
            print(f"Turno: {step_count}")
            print("Agente (X) pensando...")
            time.sleep(1.5) 
            
            valid_moves = [i for i in range(BOARD_SIZE**2) if env.is_valid_move(i)]
            
            if not valid_moves:
                print("Empate! Tabuleiro cheio.")
                break
                
            action = agent.choose_action(state_matrix, valid_moves)
            
            next_state_flat, reward, done, info = env.step(action)
            state_matrix = env.board
            
            step_count += 1
            
        print_board(env.board)
        
        if info.get('result') == 'Win':
            print("🏆 RESULTADO: O AGENTE VENCEU! 🤖")
        elif info.get('result') == 'Loss':
            print("💀 RESULTADO: O Agente perdeu.")
        else:
            print("😐 RESULTADO: Empate.")
            
        print("\nPróxima partida em 3 segundos...")
        time.sleep(3)

if __name__ == "__main__":
    play_demonstration()