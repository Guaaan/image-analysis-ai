from dotenv import load_dotenv
import os
import sys
import cv2
from datetime import datetime  # Importación añadida
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from ultralytics import YOLO

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise ValueError("Please set AI_SERVICE_ENDPOINT and AI_SERVICE_KEY in .env file")

        # Cargar modelo YOLO
        yolo_model = YOLO('yolov8n.pt')

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=str(ai_endpoint),
            credential=AzureKeyCredential(str(ai_key))
        )

        # Opción 1: Procesar imagen de archivo
        if len(sys.argv) > 1:
            image_file = sys.argv[1]
            with open(image_file, "rb") as f:
                image_data = f.read()
            AnalyzeImage(image_file, image_data, cv_client)
        
        # Opción 2: Procesar video de cámara
        else:
            process_camera(yolo_model, cv_client)

    except Exception as ex:
        print(f"Error: {ex}")

def process_camera(yolo_model, cv_client):
    """Captura video de la cámara y aplica YOLO y Azure Computer Vision"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar con YOLO
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()
        
        cv2.imshow('YOLO Detection', annotated_frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_data = img_encoded.tobytes()
            
            # Generar nombre único basado en timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"camera_capture_{timestamp}_results.txt"
            
            AnalyzeImage(output_filename, image_data, cv_client)
            
            # Opcional: guardar también la imagen
            cv2.imwrite(f"camera_capture_{timestamp}.jpg", frame)
            cv2.imshow('Original Frame', frame)
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')
    
    try:
        with open(image_filename, 'w', encoding='utf-8') as f:
            f.write(f"Análisis de imagen\n")
            f.write("===================================\n\n")
            
            result = cv_client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.PEOPLE],
            )

            if result.caption is not None:
                f.write("\nCAPTION PRINCIPAL:\n")
                f.write(f"Texto: {result.caption.text}\n")
                f.write(f"Confianza: {result.caption.confidence * 100:.2f}%\n")

            if result.dense_captions is not None:
                f.write("\nCAPTIONES DETALLADAS:\n")
                for i, caption in enumerate(result.dense_captions.list, 1):
                    f.write(f"{i}. {caption.text} (Confianza: {caption.confidence * 100:.2f}%)\n")

            if result.tags is not None:
                f.write("\nETIQUETAS:\n")
                for tag in result.tags.list:
                    f.write(f"- {tag.name} (Confianza: {tag.confidence * 100:.2f}%)\n")

            if result.objects is not None:
                f.write("\nOBJETOS:\n")
                for obj in result.objects.list:
                    f.write(f"- {obj.tags[0].name} (Confianza: {obj.tags[0].confidence * 100:.2f}%)\n")

        print(f"Resultados guardados en {image_filename}")

    except HttpResponseError as e:
        print(f"Error de Azure: {e.error.message}")
    except Exception as e:
        print(f"Error durante el análisis: {e}")

if __name__ == "__main__":
    main()