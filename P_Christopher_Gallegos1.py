import cv2
import torch
import numpy as np
from sort import Sort
import matplotlib.path as mplPath
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import threading

def generate_zones(screen_width, screen_height, num_zones):
    zones = []
    zone_width = screen_width // num_zones
    zone_height = screen_height // num_zones

    for i in range(num_zones):
        for j in range(num_zones):
            zone = np.array([
                [i * zone_width, j * zone_height],
                [i * zone_width + zone_width, j * zone_height],
                [i * zone_width + zone_width, j * zone_height + zone_height],
                [i * zone_width, j * zone_height + zone_height],
            ])
            center = (i * zone_width + zone_width // 2, j * zone_height + zone_height // 2)
            zones.append((zone, center))

    return zones

def get_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

def load_model():
    model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
    return model

def get_bboxes(preds):
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.50]
    df = df[df["name"] == "person"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

def is_valid_detection(xc, yc, zone):
    return mplPath.Path(zone).contains_point((xc, yc))

def detector(cap, output_label, zones, paused):
    model = load_model()
    tracker = Sort()

    while cap.isOpened():
        if not paused.is_set():
            status, frame = cap.read()
            if not status:
                break

            preds = model(frame)
            bboxes = get_bboxes(preds)

            pred_confidences = preds.xyxy[0][:, 4].cpu().numpy()

            trackers = tracker.update(bboxes)

            detections_zones = [0] * len(zones)

            for i, box in enumerate(trackers):
                xc, yc = get_center(box)
                xc, yc = int(xc), int(yc)

                cv2.rectangle(img=frame, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(255, 0, 0), thickness=2)  # Cambiar el color aquí
                cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0, 255, 0), thickness=-1)  # Cambiar el color aquí
                cv2.putText(img=frame, text=f"id: {int(box[4])}, conf: {pred_confidences[i]:.2f}", org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 255), thickness=2)

                for j, (zone, _) in enumerate(zones):
                    if is_valid_detection(xc, yc, zone):
                        detections_zones[j] += 1

            for j, (zone, center) in enumerate(zones):
                cv2.putText(img=frame, text=f"Area {j+1}: {detections_zones[j]}", org=(center[0] - 30, center[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.polylines(img=frame, pts=[zone], isClosed=True, color=(0, 0, 255), thickness=3)

            # Convertir el frame a formato adecuado para mostrar en un widget Label de tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            # Actualizar la imagen en el widget Label
            output_label.config(image=frame)
            output_label.image = frame

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()

def iniciar_programa(output_label):
    global programa_thread
    global cap
    global paused
    programa_thread = threading.Thread(target=detector, args=(cap, output_label, zones, paused))
    programa_thread.start()
    btn_iniciar.config(state=tk.DISABLED, bg="#808080", activebackground="#808080")
    btn_pausar.config(state=tk.NORMAL, bg="#FFA500", activebackground="#FFA500")
    btn_detener.config(state=tk.NORMAL, bg="#FF0000", activebackground="#FF0000")
    btn_reiniciar.config(state=tk.NORMAL, bg="#4682B4", activebackground="#87CEFA")

def pausar_programa():
    paused.set()
    btn_pausar.config(state=tk.DISABLED, bg="#808080", activebackground="#808080")
    btn_reanudar.config(state=tk.NORMAL, bg="#00BFFF", activebackground="#00BFFF")

def reanudar_programa():
    paused.clear()
    btn_pausar.config(state=tk.NORMAL, bg="#FFA500", activebackground="#FFA500")
    btn_reanudar.config(state=tk.DISABLED, bg="#808080", activebackground="#808080")

def reiniciar_programa():
    global cap
    cap.release()
    cap = cv2.VideoCapture("D:/Chris/Descargas/Proyeto/video.webm")
    iniciar_programa(output_label)

def detener_programa():
    cap.release()
    root.quit()

# Crear la interfaz gráfica
root = tk.Tk()
root.title("DETECTOR DE PERSONAS")
root.geometry("800x600")

# Estilos
font_style = ("Helvetica", 12)

# Marco para los botones
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

# Botones
btn_iniciar = tk.Button(button_frame, text="Iniciar", command=lambda: iniciar_programa(output_label), font=font_style, bg="#008000", activebackground="#00FF00")
btn_iniciar.pack(side=tk.LEFT, padx=5)

btn_pausar = tk.Button(button_frame, text="Pausar", command=pausar_programa, font=font_style, bg="#FFA500", activebackground="#FFA500", state=tk.DISABLED)
btn_pausar.pack(side=tk.LEFT, padx=5)

btn_reanudar = tk.Button(button_frame, text="Reanudar", command=reanudar_programa, font=font_style, bg="#808080", activebackground="#808080", state=tk.DISABLED)
btn_reanudar.pack(side=tk.LEFT, padx=5)

btn_reiniciar = tk.Button(button_frame, text="Reiniciar", command=reiniciar_programa, font=font_style, bg="#4682B4", activebackground="#87CEFA", state=tk.DISABLED)
btn_reiniciar.pack(side=tk.LEFT, padx=5)

btn_detener = tk.Button(button_frame, text="Salir", command=detener_programa, font=font_style, bg="#FF0000", activebackground="#FF0000", state=tk.DISABLED)
btn_detener.pack(side=tk.LEFT, padx=5)

# Variables
paused = threading.Event()

# Inicializar la captura de video
cap = cv2.VideoCapture("D:/Chris/Descargas/Proyeto/video.webm")

# Generar zonas
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
zones = generate_zones(screen_width, screen_height, 3)

# Etiqueta para mostrar la salida de la cámara
output_label = tk.Label(root)
output_label.pack(pady=10)

root.mainloop()
