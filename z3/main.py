import argparse
import json
import numpy as np
import sys
from operator import attrgetter

"""
Opis problemu
-------------

Z listy filmów i ich ocen dla użytkowników, wybierz pięć filmów, które poleci
i nie poleci dla danego użytkownika.

Implementacja:
- Sylwester Kąkol
- Adam Jurkiewicz
"""


def get_common_movies(dataset, user1, user2):
    """
    Pobierz wspólne filmy od dwóch użytkowników z dostarczonych danych

    Parametry
    ----------

    dataset:
      słownik zawierający dane z formatu JSON
    user1:
      pierwszy użytkownik do porównania
    user2:
      drugi użytkownik do porównania

    Zwraca
    ---------

    common_movies:
      zestaw filmów
    """
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    return common_movies


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    """
    Pobierz wspólne filmy od dwóch użytkowników z dostarczonych danych

    Parametry
    ----------

    dataset:
      słownik zawierający dane z formatu JSON
    user1:
      pierwszy użytkownik do porównania
    user2:
      drugi użytkownik do porównania

    Zwraca
    ---------

    wynik dystansu Euklidesa (Pomiędzy 0.0 - 1.0)
    """

    # If there are no common movies between the users,
    # then the score is 0
    if len(get_common_movies(dataset, user1, user2)) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearson_score(dataset, user1, user2):
    """
    Oblicz wynik Korelacji Pearsona pomiędzy użytkownikami user1 i user2

    Parametry
    ----------

    dataset:
      słownik zawierający dane z formatu JSON
    user1:
      pierwszy użytkownik do obliczeń
    user2:
      drugi użytkownik do obliczeń

    Zwraca
    ---------

    wynik korelacji Pearsona
    """
    # Movies rated by both user1 and user2
    common_movies = get_common_movies(dataset, user1, user2)
    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


class Record:
    """
    Klasa rekord, która trzyma informacje o użytkowniku i jego wyniku
    z obliczeń algorytmu Euklidesa lub korelacji Pearsona.

    Pola
    ----------

    name:
      Imię i nazwisko użytkownika
    score:
      Wynik obliczeń
    """

    def __init__(self, user, score):
        self.name = user
        self.score = score

    def __repr__(self):
        return f"{self.name}: {self.score}"


def print_selections(user_data, excluded_set, should_reverse):
    """
    Wypisuje do pięciu filmów, które są posortowane według najlepszych,
    lub najgorszych dla danego użytkownika.

    Parametry
    ----------

    user_data:
      filmy należące do danego użytkownika
    excluded_set:
      zestaw filmów, które mają być wykluczone z user_data
    should_reverse:
      boolean dla funkcji sorted(); wartość True dla najlepiej ocenianych
      filmów, wartość False dla najgorzej ocenianych filmów
    """
    movie_set = sorted(user_data, key=data[selected_user.name].get, reverse=should_reverse)
    selected_amount = 0
    for title in movie_set:
        if selected_amount > 4:
            break
        if title not in excluded_set:
            print(title)
            selected_amount += 1


if __name__ == '__main__':
    user = input("Enter name: ")
    score_type = input("Enter score type [Pearson, Euclidean]: ")
    # target user is the user that program will propose movies
    target_user = user
    # user_list is a list will all users in the dataset EXCEPT target_user
    user_list = []
    scores = []
    ratings_file = 'pjatk-ratings.json'

    with open(ratings_file, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())

    # check if target_user is in given set
    if target_user not in data:
        raise TypeError('Cannot find ' + target_user + ' in the dataset')

    # fill user_list
    for user in data:
        if not user == target_user:
            user_list.append(user)

    # iterate over user_list to fill scores array
    # possible improvement for future to not even store users with score 0, these are useless
    for user in user_list:
        if score_type == 'Euclidean':
            scores.append(Record(user, euclidean_score(data, target_user, user)))
        else:
            scores.append(Record(user, pearson_score(data, target_user, user)))

    selected_user = max(scores, key=attrgetter('score'))
    movies_to_exclude = get_common_movies(data, target_user, selected_user.name)

    print(f"{score_type} magic predicts you will enjoy these movies:")
    print_selections(data[selected_user.name], movies_to_exclude, True)
    print(f"{score_type} magic also predicts that you shouldn't watch these movies:")
    print_selections(data[selected_user.name], movies_to_exclude, False)
