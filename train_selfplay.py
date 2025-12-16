# train_selfplay.py
import numpy as np
import time
import os
import pickle
from environment import TicTacToeEnv
from agent import QAgent
from settings import *

def train_self_play():
    print("⚔️ PREPARANDO ARENA DE AUTO-APERFEIÇOAMENTO ⚔️")
   
    teacher_agent = QAgent()
    if os.path.exists("brain.pkl"):
        print("✅ Conhecimento anterior carregado para os oponentes.")
        teacher_agent.load_model("brain.pkl")
        teacher_agent.epsilon = 0.0  
        teacher_agent.alpha = 0.0    
    else:
        print("❌ ERRO: brain.pkl não encontrado! Treine o básico primeiro.")
        return

    student_agent = QAgent()
    student_agent.load_model("brain.pkl")
    student_agent.epsilon = 0.3  
    student_agent.alpha = 0.1    

    env = TicTacToeEnv(opponent_brain=teacher_agent)
    
    print(f"🎯 Meta: {EPISODES} episódios contra 3 cópias do Agente Anterior.")
    print("-" * 50)
    
    start_time = time.time()
    recent_wins = [] 
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        state_matrix = env.board.copy()
        done = False
        
        while not done:
            valid_moves = [i for i in range(BOARD_SIZE * BOARD_SIZE) if env.is_valid_move(i)]
            if not valid_moves: break

            action = student_agent.choose_action(state_matrix, valid_moves)
            
            next_state_flat, reward, done, info = env.step(action)
            next_state_matrix = env.board.copy()

            student_agent.learn(state_matrix, action, reward, next_state_matrix)
            state_matrix = next_state_matrix

        if info.get('result') == 'Win':
            recent_wins.append(1)
        else:
            recent_wins.append(0) 
            
        if len(recent_wins) > 1000: recent_wins.pop(0)

        student_agent.decay_alpha()
        if student_agent.epsilon > 0.01:
            student_agent.epsilon *= 0.99995

        if episode % 1000 == 0:
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            print(f"Episódio {episode:6d} | Win Rate: {win_rate:5.1f}% | Epsilon: {student_agent.epsilon:.3f} | Q-Size: {len(student_agent.q_table)}")

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ Treino Self-Play finalizado em {total_time:.1f}s")
    
    student_agent.save_model("brain_v2_elite.pkl")
    print("💾 Novo modelo salvo como 'brain_v2_elite.pkl'")

if __name__ == "__main__":
    train_self_play()