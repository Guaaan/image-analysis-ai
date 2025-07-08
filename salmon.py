from dotenv import load_dotenv
import os
import sys
import cv2
from datetime import datetime
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from ultralytics import YOLO
import numpy as np

def main():
    try:
        # Configuración inicial
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise ValueError("Please set AI_SERVICE_ENDPOINT and AI_SERVICE_KEY in .env file")

        # Cargar modelo YOLO
        yolo_model = YOLO('yolov8n.pt')  # Asegúrate que el modelo puede detectar peces

        # Cliente Azure AI Vision
        cv_client = ImageAnalysisClient(
            endpoint=str(ai_endpoint),
            credential=AzureKeyCredential(str(ai_key))
            )
        
        # Procesar video o cámara
        if len(sys.argv) > 1:
            process_video(sys.argv[1], yolo_model, cv_client)
        else:
            process_camera(yolo_model, cv_client)

    except Exception as ex:
        print(f"Error: {ex}")

def process_video(video_path, yolo_model, cv_client):
    """Procesa un video detectando y analizando peces"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return
    
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nProcesando video: {video_path}")
    print("Presiona 's' para capturar frame actual")
    print("Presiona 'q' para salir...\n")
    
    frame_count = 0
    capture_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detección con YOLO
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()
        
        # Mostrar frame con detecciones
        cv2.imshow('Detección de Peces', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Captura manual
            capture_count += 1
            print(f"\nCapturando frame {frame_count} manualmente...")
            
            # Guardar imagen completa
            img_path = f"{results_dir}/capture_{capture_count}.jpg"
            cv2.imwrite(img_path, frame)
            
            # Analizar imagen
            with open(img_path, "rb") as img_file:
                image_data = img_file.read()
            
            output_filename = f"{results_dir}/analysis_{capture_count}.txt"
            AnalyzeImage(output_filename, image_data, cv_client)
            print(f"Análisis guardado en {output_filename}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nProcesamiento de video completado")

def process_camera(yolo_model, cv_client):
    """Procesa video de cámara detectando y analizando peces"""
    cap = cv2.VideoCapture(0)
    
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nCámara iniciada - Detectando peces...")
    print("Presiona 's' para capturar frame actual")
    print("Presiona 'q' para salir...\n")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detección con YOLO
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()
        
        # Mostrar frame con detecciones
        cv2.imshow('Detección de Peces en Tiempo Real', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Captura manual
            capture_count += 1
            print("\nCapturando frame manualmente...")
            
            # Guardar imagen completa
            img_path = f"{results_dir}/capture_{capture_count}.jpg"
            cv2.imwrite(img_path, frame)
            
            # Analizar imagen
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_data = img_encoded.tobytes()
            
            output_filename = f"{results_dir}/analysis_{capture_count}.txt"
            AnalyzeImage(output_filename, image_data, cv_client)
            print(f"Análisis guardado en {output_filename}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nProcesamiento de cámara completado")

def AnalyzeImage(image_filename, image_data, cv_client):
    """Analiza imagen con Azure AI Vision y guarda resultados"""
    try:
        with open(image_filename, 'w', encoding='utf-8') as f:
            f.write("Análisis de Imagen\n")
            f.write("===================================\n\n")
            
            result = cv_client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS],
            )

            if result.caption is not None:
                f.write("\nDESCRIPCIÓN:\n")
                f.write(f"{result.caption.text} (Confianza: {result.caption.confidence * 100:.2f}%)\n")

            if result.tags is not None:
                f.write("\nETIQUETAS RELEVANTES:\n")
                for tag in result.tags.list:
                    if tag.confidence > 0.5:
                        f.write(f"- {tag.name} (Confianza: {tag.confidence * 100:.2f}%)\n")

            if result.objects is not None:
                f.write("\nOBJETOS DETECTADOS:\n")
                for obj in result.objects.list:
                    if obj.tags[0].confidence > 0.5:
                        f.write(f"- {obj.tags[0].name} (Confianza: {obj.tags[0].confidence * 100:.2f}%)\n")

        print(f"Resultados guardados en {image_filename}")

    except HttpResponseError as e:
        print(f"Error de Azure: {e.error.message}")
    except Exception as e:
        print(f"Error durante el análisis: {e}")

if __name__ == "__main__":
    main()