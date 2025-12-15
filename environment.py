# environment.py
import numpy as np
import random
from settings import *

class TicTacToeEnv:
    def __init__(self, opponent_brain=None):
        """
        opponent_brain: Instância de QAgent já treinada (congelada).
        Se for None, os oponentes jogam aleatoriamente.
        """
        self.opponent_brain = opponent_brain
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()

    def is_valid_move(self, action):
        row, col = divmod(action, BOARD_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row, col] == EMPTY
        return False

    def check_winner(self, player_id):
        # ... (MANTENHA A MESMA LÓGICA DE VITÓRIA QUE JÁ FUNCIONA) ...
        # (Vou resumir aqui para economizar espaço, use o seu código atual de check_winner)
        b = self.board
        n = BOARD_SIZE
        target = WIN_LENGTH
        # Linhas/Colunas
        for i in range(n):
            row = b[i, :]; col = b[:, i]
            for j in range(n - target + 1):
                if np.all(row[j:j+target] == player_id): return True
                if np.all(col[j:j+target] == player_id): return True
        # Diagonais
        for r in range(n - target + 1):
            for c in range(n - target + 1):
                sub = b[r:r+target, c:c+target]
                if np.all(sub.diagonal() == player_id): return True
                if np.all(np.fliplr(sub).diagonal() == player_id): return True
        return False

    def is_draw(self):
        return not np.any(self.board == EMPTY)

    def get_opponent_view(self, player_id):
        """
        Transforma o tabuleiro para que o 'player_id' se veja como o '1' (Agente).
        E veja todos os outros (incluindo o agente original) como inimigos.
        """
        view = self.board.copy()
        
        # 1. Identificar onde é o jogador atual e marcar temporariamente
        my_pos = (view == player_id)
        
        # 2. Todo o resto que não é vazio vira '2' (Inimigo Genérico)
        # Isso transforma o Agente(1) e os outros bots em Inimigos(2)
        others_pos = (view != 0) & (view != player_id)
        view[others_pos] = 2 
        
        # 3. Transformar o jogador atual em '1' (Para bater com a Q-Table)
        view[my_pos] = 1
        
        return view # Retorna matriz 4x4 (o agent.choose_action espera matriz)

    def play_opponents(self):
        if self.done: return
        
        for opp_id in OPPONENTS:
            if self.is_draw(): self.done = True; return

            # Pega as jogadas válidas
            valid_moves = [i for i in range(BOARD_SIZE**2) if self.board.flatten()[i] == 0]
            if not valid_moves: self.done = True; return

            # --- DECISÃO: CÉREBRO OU ALEATÓRIO? ---
            if self.opponent_brain:
                # Cria a ilusão de ótica para o bot
                opp_view = self.get_opponent_view(opp_id)
                
                # O bot escolhe a melhor ação baseada no treino anterior
                # Nota: O epsilon do opponent_brain deve ser 0 (sem aleatoriedade)
                action = self.opponent_brain.choose_action(opp_view, valid_moves)
            else:
                action = random.choice(valid_moves)
            # --------------------------------------

            row, col = divmod(action, BOARD_SIZE)
            self.board[row, col] = opp_id

            if self.check_winner(opp_id):
                self.winner = opp_id
                self.done = True
                return
            
            if self.is_draw(): self.done = True; return

    def step(self, action):
        # ... (MANTENHA O MESMO STEP QUE VOCÊ JÁ TEM, ELE CHAMA O play_opponents ACIMA) ...
        # Apenas certifique-se de que ele chama self.play_opponents() no final.
        if self.done: return self.board.flatten(), 0, True, {}
        
        if not self.is_valid_move(action):
            return self.board.flatten(), REWARDS['INVALID'], self.done, {}

        # Executa Jogada do Agente Principal
        row, col = divmod(action, BOARD_SIZE)
        self.board[row, col] = AGENT_ID
        
        if self.check_winner(AGENT_ID):
            self.done = True
            return self.board.flatten(), REWARDS['WIN'], True, {'result': 'Win'}
            
        reward = REWARDS['STEP']
        if self.is_draw():
            self.done = True
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        # Oponentes jogam (AGORA USANDO O CÉREBRO SE ESTIVER CONFIGURADO)
        self.play_opponents()

        if self.done:
            if self.winner in OPPONENTS:
                return self.board.flatten(), REWARDS['LOSS'], True, {'result': 'Loss'}
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        return self.board.flatten(), reward, False, {}