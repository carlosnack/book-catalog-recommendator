import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Carrega o dataset de livros
books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
users = pd.read_csv("BX-Users.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", on_bad_lines='skip')

# Seleciona e renomeia colunas importantes do DataFrame de livros
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'}, inplace=True)

# Seleciona e renomeia colunas relevantes do DataFrame de avaliações
ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Seleciona e renomeia colunas importantes do DataFrame de usuários
users = users[['User-ID', 'Location', 'Age']]
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)

# Conta o número de avaliações feitas por cada usuário
user_ratings_count = ratings['user_id'].value_counts()

# Filtra usuários com mais de 200 avaliações
frequent_users = user_ratings_count[user_ratings_count > 200]

# Exibe o número de usuários frequentes
print(frequent_users.shape)

# Mantém avaliações apenas dos usuários frequentes
ratings = ratings[ratings['user_id'].isin(frequent_users.index)]

# Combina as avaliações com as informações dos livros
ratings_with_books = ratings.merge(books, on='ISBN')

# Conta o número de avaliações por livro
book_ratings_count = ratings_with_books.groupby('title')['rating'].count().reset_index()

# Renomeia a coluna de contagem de avaliações
book_ratings_count.rename(columns={'rating': 'number_of_ratings'}, inplace=True)

# Combina a contagem de avaliações com o DataFrame principal
final_ratings = ratings_with_books.merge(book_ratings_count, on='title')

# Filtra para manter apenas livros com pelo menos 50 avaliações
popular_books = final_ratings[final_ratings['number_of_ratings'] >= 50]
popular_books.drop_duplicates(['user_id', 'title'], inplace=True)

#Exibe a cobertura das recomendações
coverage = len(set(popular_books['title'])) / len(set(books['title']))
print(f'Coverage: {coverage * 100:.2f}%')

# Cria uma tabela pivô de avaliações de livros por usuário
book_user_matrix = popular_books.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

# Converte a tabela pivô para uma matriz esparsa
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(book_user_matrix)

# Inicializa o modelo de vizinhos mais próximos
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)


# Testa o modelo para encontrar os vizinhos mais próximos de um livro específico
distances, suggestions = model.kneighbors(book_user_matrix.iloc[238, :].values.reshape(1, -1))


# Salvar o modelo, a matriz e os dados em arquivos pickle na raiz do diretório
with open('book_recommender_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('book_user_matrix.pkl', 'wb') as matrix_file:
    pickle.dump(book_user_matrix, matrix_file)

with open('book_titles.pkl', 'wb') as titles_file:
    pickle.dump(book_user_matrix.index.tolist(), titles_file)

