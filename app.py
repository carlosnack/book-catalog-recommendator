import streamlit as st
import pandas as pd

# Carrega o dataset de livros
books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
users = pd.read_csv("BX-Users.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", on_bad_lines='skip')

import pickle

# Carregar o modelo, a matriz e os títulos dos livros da raiz do diretório
with open('book_recommender_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('book_user_matrix.pkl', 'rb') as matrix_file:
    book_user_matrix = pickle.load(matrix_file)

with open('book_titles.pkl', 'rb') as titles_file:
    book_titles = pickle.load(titles_file)

# Função para recomendar livros
def recommend_books(book_title, n_recommendations=5):
    book_index = book_titles.index(book_title)
    distances, suggestions = model.kneighbors(book_user_matrix.iloc[book_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    recommendations = []
    for suggestion in suggestions[0]:
        if suggestion != book_index:  # Excluir o próprio livro da lista de recomendações
            recommendations.append(book_titles[suggestion])
    return recommendations


# Seleciona e renomeia colunas importantes do DataFrame de livros
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
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

import streamlit as st
from PIL import Image

# Inicializar estado para navegação
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'book_selected' not in st.session_state:
    st.session_state.book_selected = None

# Função para exibir a página inicial
def home_page():
    st.title("Catálogo de Livros - Recomendações Personalizadas")
    st.subheader("Descubra livros incríveis e veja recomendações baseadas em suas preferências!")

    # Exibir os livros mais populares
    most_popular = popular_books.groupby('title').agg({'rating': 'mean', 'number_of_ratings': 'max'}).reset_index()
    most_popular = most_popular.sort_values(by='number_of_ratings', ascending=False).head(30)

    st.write("### Livros Mais Populares:")


    # Criar uma grade de livros em formato de cards
    columns = st.columns(3)  
    for idx, row in most_popular.iterrows():
        col = columns[idx % 3]  # Alterna entre as colunas
        with col:
            # Truncar título para evitar desalinhamento
            truncated_title = row['title'][:30] + '...' if len(row['title']) > 30 else row['title']
            
            image_url = books[books['title'] == row['title']]['Image-URL-L'].values[0]

            st.image(image_url if pd.notna(image_url) else "https://via.placeholder.com/150", caption=truncated_title, width=150)
            
            # Mostrar avaliação e número de avaliações
            st.markdown(f"⭐ **{row['rating']:.2f}** ({row['number_of_ratings']} avaliações)")
            
            # Botão "Ver Detalhes" com chave única
            if st.button("Ver Detalhes", key=f"details_{idx}"):
                st.session_state.book_selected = row['title']
                st.session_state.page = 'details'

# Função para exibir a página de detalhes
def details_page():
    if st.session_state.book_selected:
        book_selected = st.session_state.book_selected
        st.button("Voltar", on_click=lambda: setattr(st.session_state, 'page', 'home'))

        st.write(f"### Detalhes do Livro: {book_selected}")
        book_info = books[books['title'] == book_selected].iloc[0]
        st.write(f"- **Autor:** {book_info['author']}")
        st.write(f"- **Ano de Publicação:** {book_info['year']}")
        st.write(f"- **Editora:** {book_info['publisher']}")

        st.write("### Recomendações Baseadas no Livro Selecionado:")
        recommendations = recommend_books(book_selected)
        
        # Exibir recomendações em formato de cards
        rec_columns = st.columns(5)
        for rec_idx, rec_title in enumerate(recommendations):
            rec_col = rec_columns[rec_idx % 5]
            with rec_col:
                st.image("https://via.placeholder.com/150", caption=rec_title, use_container_width=True)
                avg_rating = popular_books[popular_books['title'] == rec_title]['rating'].mean()
                st.markdown(f"⭐ **{avg_rating:.2f}**")

# Controlar a navegação entre páginas
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'details':
    details_page()