from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import customtkinter as ctk
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import traceback
import joblib
import os
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    frm_width = window.winfo_rootx() - window.winfo_x()
    win_width = width + 2*frm_width
    height = window.winfo_height()
    titlebar_height = window.winfo_rooty() - window.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = window.winfo_screenwidth()//2 - win_width//2
    y = window.winfo_screenheight()//2 - win_height//2
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    window.deiconify()


class TextClassifier:
    def __init__(self):
        # Categorías predefinidas
        self.categories = ['Noticias', 'Deportes', 'Tecnología', 'Entretenimiento', 'Ciencia', 'Salud','Recetas']
        
        # Datos de entrenamiento de ejemplo
        train_texts = [
            "Hubo un accidente en la ruta",
            "El presidente presentó nuevas políticas económicas",
            "Se realizará la cumbre internacional de economía",
            "Cristiano Ronaldo marcó tres goles en el partido",
            "La selección nacional entrenó para el próximo mundial",
            "Apple lanzó un nuevo iPhone con inteligencia artificial",
            "Hollywood prepara nueva película de superhéroes",
            "Ganadores de los premios de cine fueron anunciados",
            "Los científicos descubrieron una nueva partícula subatómica",
            "Se publicó un estudio sobre el cambio climático",
            "Los médicos recomiendan caminar 30 minutos al día",
            "Nueva vacuna contra la gripe muestra alta eficacia",
            "Receta de pastel de chocolate fácil y rápido",
            "Cómo preparar una ensalada César clásica",
            "Paso a paso para hacer pizza casera",
            "Ingredientes para una sopa de verduras saludable"
        ]
        
        train_labels = [
            'Noticias', 'Noticias', 
            'Deportes', 'Deportes', 
            'Tecnología', 'Tecnología', 
            'Entretenimiento', 'Entretenimiento',
            'Ciencia', 'Ciencia', 
            'Salud', 'Salud',
            'Recetas', 'Recetas',
            'Recetas', 'Recetas'
        ]

        # Crear pipeline de clasificación
        self.classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('clf', MultinomialNB())
        ])
        
        # Entrenar el modelo
        self.classifier.fit(train_texts, train_labels)

    def predict_category(self, text):
        """Predecir la categoría de un texto"""
        return self.classifier.predict([text])[0]

class TextProcessorApp:
    def __init__(self):
        # Historial de entradas procesadas
        self.history = []
        self.text_classifier = TextClassifier()

        # Configurar la ventana principal
        self.window = ctk.CTk()
        self.window.title("Procesador de Texto")
        self.window.geometry("1200x800")  # Ventana más ancha para acomodar el diseño lado a lado
        self.window.resizable(0,0)
        
        # Centrar la ventana después de un breve delay
        self.window.after(100, lambda: center_window(self.window))

        # Frame principal
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Título
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Procesador de Texto - Análisis y Resumen Automático",
            font=("Arial", 20,"bold")
        )
        self.title_label.pack(pady=10)

        # Frame para contener los dos paneles lado a lado
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Panel izquierdo
        self.left_panel = ctk.CTkFrame(self.content_frame)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=5)

        # Panel derecho
        self.right_panel = ctk.CTkFrame(self.content_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=5)

        # Área de texto de entrada (panel izquierdo)
        self.input_label = ctk.CTkLabel(
            self.left_panel,
            text="Texto de entrada:",
            font=("Arial", 18,"bold")
        )
        self.input_label.pack(pady=(10,5))

        self.input_text = ctk.CTkTextbox(
            self.left_panel,
            height=500,  # Altura aumentada
            font=("Arial", 14)
        )
        self.input_text.pack(padx=10, pady=(0,10), fill="both", expand=True)

        # Área de resultados (panel derecho)
        self.result_label = ctk.CTkLabel(
            self.right_panel,
            text="Resultados:",
            font=("Arial", 18,"bold")
        )
        self.result_label.pack(pady=(10,5))

        self.result_text = ctk.CTkTextbox(
            self.right_panel,
            height=500,  # Altura aumentada
            font=("Arial", 14)
        )
        self.result_text.pack(padx=10, pady=(0,10), fill="both", expand=True)

        # Frame para controles en la parte inferior
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=5)

        # Control de número de párrafos para el resumen
        self.summary_label = ctk.CTkLabel(
            self.controls_frame,
            text="Número de párrafos para el resumen:",
            font=("Arial", 14,"bold")
        )
        self.summary_label.pack(side="left", padx=(10,10))

        self.sentences_var = ctk.StringVar(value="3")
        self.sentences_entry = ctk.CTkEntry(
            self.controls_frame,
            width=50,
            textvariable=self.sentences_var
        )
        self.sentences_entry.pack(side="left", padx=10)

        # Frame para botones
        self.buttons_frame = ctk.CTkFrame(self.main_frame)
        self.buttons_frame.pack(fill="x", padx=10, pady=10)

        # Botones
        self.process_button = ctk.CTkButton(
            self.buttons_frame,
            text="Procesar Texto",
            command=self.process_text,
            font=("Arial", 12,"bold"),
            width=150
        )
        self.process_button.pack(side="left", padx=5)

        self.summarize_button = ctk.CTkButton(
            self.buttons_frame,
            text="Generar Resumen",
            command=self.generate_summary,
            font=("Arial", 12,"bold"),
            width=150
        )
        self.summarize_button.pack(side="left", padx=5)

        self.clear_button = ctk.CTkButton(
            self.buttons_frame,
            text="Borrar Resultado",
            command=self.clear_result,
            font=("Arial", 12, "bold"),
            width=150
        )
        self.clear_button.pack(side="left", padx=5)

        self.history_button = ctk.CTkButton(
            self.buttons_frame,
            text="Mostrar Historial",
            command=self.show_history,
            font=("Arial", 12,"bold"),
            width=150
        )
        self.history_button.pack(side="right", padx=5)

        self.classify_button = ctk.CTkButton(
            self.buttons_frame,
            text="Clasificar Texto",
            command=self.classify_text,
            font=("Arial", 12, "bold"),
            width=150
        )
        self.classify_button.pack(side="left", padx=5)

        # Descargar recursos de NLTK necesarios
        try:
            nltk.download("punkt")
            nltk.download("stopwords")
        except Exception as e:
            print(f"Error descargando recursos NLTK: {e}")
    
    def process_text(self):
        # Obtener el texto de entrada
        input_text = self.input_text.get("1.0", "end-1c")
        
        # Obtener stopwords en español
        spanish_sw = set(stopwords.words('spanish'))
        
        # Tokenizar y filtrar stopwords
        filtered_text = []
        words = word_tokenize(input_text)
        
        for word in words:
            if word.lower() not in spanish_sw:
                filtered_text.append(word)
        
        # Análisis de sentimiento
        analysis = TextBlob(input_text)
        sentiment = analysis.sentiment.polarity
        if sentiment > 0:
            sentiment_text = "Positivo"
        elif sentiment < 0:
            sentiment_text = "Negativo"
        else:
            sentiment_text = "Neutral"
        
        # Almacenar en el historial
        self.history.append((input_text, filtered_text, sentiment_text))

        # Mostrar resultado
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", f"TEXTO FILTRADO: {str(filtered_text)}\n\n")
        self.result_text.insert("end", f"SENTIMIENTO DE TEXTO: {sentiment_text}")

    def generate_summary(self):
        #"""Genera un resumen del texto usando LSA Summarizer"""
        try:
            input_text = self.input_text.get("1.0", "end-1c")
            
            # Verificar que el texto tenga suficiente contenido
            sentences = sent_tokenize(input_text)
            if len(sentences) < 3:
                self.result_text.delete("1.0", "end")
                self.result_text.insert("1.0", "Error: El texto es demasiado corto para generar un resumen. Se necesitan al menos 3 oraciones.")
                return

            # Configurar el summarizer
            parser = PlaintextParser.from_string(input_text, Tokenizer("spanish"))
            stemmer = Stemmer("spanish")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("spanish")

            # Obtener el número de oraciones del entry
            try:
                num_sentences = min(int(self.sentences_var.get()), len(sentences))
            except ValueError:
                num_sentences = 3

            # Generar el resumen
            summary_sentences = summarizer(parser.document, num_sentences)
            summary = "\n".join([str(sentence) for sentence in summary_sentences])
            
            if not summary:
                summary = "No se pudo generar un resumen. El texto puede ser demasiado corto o poco coherente."

            # Mostrar el resumen
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "RESUMEN AUTOMÁTICO:\n\n")
            self.result_text.insert("end", summary)
            
            # Agregar al historial
            self.history.append((input_text, "RESUMEN", summary))

        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error al generar el resumen: {str(e)}\n")
            traceback.print_exc()
        
    def train_text_classifier(self):
        """Entrena un modelo simple de clasificación de texto."""
        # Datos de ejemplo (puedes sustituirlos por un dataset más grande)
        data = [
            ("El equipo ganó partido", "Deportes"),
            ("El gobierno aprobó una nueva ley", "Política"),
            ("Apple lanza un nuevo iPhone", "Tecnología"),
            ("La película fue un éxito en taquilla", "Entretenimiento"),
            ("El jugador fue transferido a otro club", "Deportes"),
            ("Se discutió el presupuesto anual en el congreso", "Política"),
            ("Microsoft anuncia actualizaciones de Windows", "Tecnología"),
            ("El actor principal ganó un premio importante", "Entretenimiento")
        ]
        texts, labels = zip(*data)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Crear pipeline con TF-IDF y Naive Bayes
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        print("Clasificación:\n", classification_report(y_test, y_pred, target_names=self.categories))
        
        return model
    
    def export_history(self):
        """Exporta el historial de análisis y clasificaciones a un archivo CSV."""
        file_path = "text_analysis_history.csv"
        try:
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Texto Original", "Tipo de Análisis", "Resultado"])
                for entry in self.history:
                    writer.writerow(entry)
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Historial exportado exitosamente a {file_path}.")
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error al exportar el historial: {str(e)}")

    def classify_text(self):
        """Clasifica el texto ingresado según categorías."""
        
            # Obtener texto de entrada
        input_text = self.input_text.get("1.0", "end-1c")
            
            # Verificar si el texto está vacío
        try:
            # Predecir categoría
            category = self.text_classifier.predict_category(input_text)
            
            # Mostrar resultado
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Categoría detectada: {category}")
            
            # Agregar al historial
            self.history.append((input_text, "CATEGORÍA", category))
        
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error al clasificar: {str(e)}")
        
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error al clasificar el texto: {str(e)}\n")
            traceback.print_exc()
        
    def show_history(self):
        # Mostrar el historial en el área de resultados
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "Historial de Procesamientos:\n\n")
        
        for i, entry in enumerate(self.history, start=1):
            self.result_text.insert("end", f"Entrada {i}:\n")
            self.result_text.insert("end", f"Texto Original: {entry[0]}\n")
            if entry[1] == "RESUMEN":
                self.result_text.insert("end", f"Resumen Generado: {entry[2]}\n")
            else:
                self.result_text.insert("end", f"Texto Filtrado: {entry[1]}\n")
                self.result_text.insert("end", f"Sentimiento: {entry[2]}\n")
            self.result_text.insert("end", "\n")
    
    def clear_result(self):
        """Borra el contenido del área de resultados."""
        self.result_text.delete("1.0", "end")

    def run(self):
        # Añadir botón para exportar historial
        self.export_button = ctk.CTkButton(
            self.buttons_frame,
            text="Exportar Historial",
            command=self.export_history,
            font=("Arial", 12, "bold"),
            width=150
        )
        self.export_button.pack(side="left", padx=5)

        # Iniciar la aplicación
        self.window.mainloop()
    
if __name__ == "__main__":
    app = TextProcessorApp()
    app.run()
