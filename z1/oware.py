from easyAI import Negamax, TwoPlayerGame, Human_Player, AI_Player


class Oware(TwoPlayerGame):
    """
    Gra [Oware](https://en.wikipedia.org/wiki/Oware)
    
    Implementacja:
    - Sylwester Kąkol
    - Adam Jurkiewicz

    Definicja nowej gry opiera się na stworzeniu instancji klasy Oware
    oraz wywołaniu metody `play()`. Klasa jest subklasą klasy
    TwoPlayersGame i dziedziczy po niej metody.
    """

    def __init__(self, players):
        """
        Metoda inicjalizująca klasę

        Parametry
        ----------
        
        players:
          Gracze biorący udział w grze, najczęściej Human_Player i AI_Player.
        """
        for i, player in enumerate(players):
            player.score = 0
            player.isStarved = False
            player.camp = i
        self.players = players
        self.board = [4 for _ in range(12)]
        self.current_player = 1 # Human player starts

    def make_move(self, move):
        """
        Funkcja wykonująca ruch.

        - W przypadku braku możliwych ruchów, gra kończy się i dodaj do wyniku gracza
        sumę swoich sześciu dołków.

        Parametry
        ---------

        move:
          wejście od gracza (litera odpowiadająca dołkowi)
        """
        if move == "None":
            self.player.isStarved = True
            startingPoint = 6 * self.opponent.camp
            self.player.score += sum(self.board[startingPoint:startingPoint + 6])
            return

        move = 'abcdefghijkl'.index(move) # np. b = 1

        pos = move # 1
        for i in range(self.board[move]):
            pos = (pos + 1) % 12 # indeks pierwszego nasionka; zaczynamy od początku, gdy dojdzie powyżej 11
            if pos == move:
                pos = (pos + 1) % 12
            self.board[pos] += 1

        self.board[move] = 0

        while (pos / 6) == self.opponent.camp and (self.board[pos] in [2, 3]):
            self.player.score += self.board[pos]
            self.board[pos] = 0
            pos = (pos - 1) % 12

    def possible_moves(self):
        """
        Funkcja definiująca dostępne ruchy (rozsiewy). W skrócie:

        - Brak dostępnych ruchów: jeśli dołki gracza nie mają nasionek.
        - Standardowe ruchy: Pozwól na ruchy, gdzie przynajmniej jedno z
        przeniesionych nasionek znajdzie się na polu przeciwnika.
        - Jeśli brakuje standardowych ruchów: pozwól na jakikolwiek ruch, gdzie
        nasionka != 0.
        """
        if self.current_player == 1:
            if max(self.board[:6]) == 0: return ['None'] # Human player moves
            moves = [i for i in range(6) if (self.board[i] >= 6 - i)] # a b c d e f
            if moves == []: # if no moves are possible
                moves = [i for i in range(6) if self.board[i] != 0] # if e. g. e is 0, don't allow the move
        else:
            if max(self.board[6:]) == 0: return ['None'] # AI player moves
            moves = [i for i in range(6, 12) if (self.board[i] >= 12 - i)] # g h i j k l
            if moves == []:
                moves = [i for i in range(6, 12) if self.board[i] != 0]

        return ['abcdefghijkl'[u] for u in moves]

    def show(self):
        """
        Funkcja wypisująca pole gry oraz punktację.
        Na brzegach znajdują się oznaczenia dołków.
        Dołki u góry należą do AI, dołki na dole do gracza.
        """
        print("Score: %d / %d" % tuple(p.score for p in self.players))
        print('  '.join('lkjihg'))
        print(' '.join(["%02d" % i for i in self.board[-1:-7:-1]]))
        print(' '.join(["%02d" % i for i in self.board[:6]]))
        print('  '.join('abcdef'))

    def lose(self):
        """
        Funkcja zwraca True, jeśli wynik gracza jest większy, niż 24 (czyli ma
        ponad połowę sadzonek).
        """
        return self.opponent.score > 24

    def is_over(self):
        """
        Funkcja sprawdza, czy gra się skończyła sprawdzając warunki brzegowe:

        - Czy gracz przegrał
        - Gracz ma 6 lub mniej sadzonek
        - Gracz nie może wykonać kolejnego ruchu (isStarved)
        """
        return self.lose() or sum(self.board) < 7 or self.opponent.isStarved


if __name__ == "__main__":
    scoring = lambda game: game.player.score - game.opponent.score
    ai = Negamax(10, scoring)
    game = Oware([Human_Player(), AI_Player(ai)])

    game.play()

    if game.player.score > game.opponent.score:
        print("Congratulations! You won.")
    else:
        print("You lost!")