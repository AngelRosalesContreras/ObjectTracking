import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import time
import os
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO


class ObjectDetectorApp:
    def __init__(self, root):
        # Configuración principal
        self.root = root
        self.root.title("Sistema de Detección de Objetos - YOLOv8")
        self.root.geometry("1200x800")

        # Tema de colores
        self.bg_color = "#1e1e2e"  # Catppuccin Mocha
        self.text_color = "#cdd6f4"
        self.accent_color = "#89b4fa"  # Azul
        self.highlight_color = "#a6e3a1"  # Verde
        self.warning_color = "#f38ba8"  # Rojo
        self.panel_color = "#313244"

        # Variables de control
        self.cap = None
        self.is_video_playing = False
        self.detection_enabled = True
        self.current_frame = None
        self.conf_threshold = 0.5
        self.source_path = None
        self.model_size = tk.StringVar(value="yolov8n")
        self.output_video = None
        self.recording = False
        self.fps_avg = 0
        self.detection_results = []
        self.is_processing = False

        # Configuración del estilo ttk
        self.setup_style()

        # Configurar la interfaz
        self.setup_ui()

        # Cargar modelo YOLO
        self.load_model()

    def setup_style(self):
        # Configurar estilos para ttk
        style = ttk.Style()
        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background=self.panel_color)

        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=self.panel_color, foreground=self.text_color)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground=self.accent_color,
                        background=self.bg_color)

        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.map("TButton", background=[("active", self.highlight_color)])

        style.configure("Accent.TButton", background=self.accent_color, foreground=self.bg_color)
        style.map("Accent.TButton", background=[("active", self.highlight_color)])

        style.configure("Horizontal.TScale", background=self.bg_color)
        style.configure("TCheckbutton", background=self.bg_color, foreground=self.text_color)

    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Detección de Objetos con YOLOv8", style="Title.TLabel")
        title_label.pack(pady=(0, 15))

        # Panel de controles
        control_frame = ttk.Frame(main_frame, style="Panel.TFrame", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 15))

        # Fuentes de entrada
        source_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        source_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(source_frame, text="Fuente:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(source_frame, text="Cámara", command=self.start_webcam).pack(side=tk.LEFT, padx=2)
        ttk.Button(source_frame, text="Video", command=self.select_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(source_frame, text="Imagen", command=self.select_image).pack(side=tk.LEFT, padx=2)

        # Selector de modelo
        model_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        model_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(model_frame, text="Modelo:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 5))

        model_options = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        model_menu = ttk.Combobox(model_frame, textvariable=self.model_size, values=model_options, width=8,
                                  state="readonly")
        model_menu.pack(side=tk.LEFT)
        model_menu.bind("<<ComboboxSelected>>", self.change_model)

        # Control de umbral de confianza
        threshold_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        threshold_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(threshold_frame, text="Confianza:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 5))

        self.conf_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, value=self.conf_threshold, length=100,
                                    command=self.update_conf_threshold)
        self.conf_scale.pack(side=tk.LEFT)

        self.conf_label = ttk.Label(threshold_frame, text=f"{self.conf_threshold:.1f}", style="Panel.TLabel")
        self.conf_label.pack(side=tk.LEFT, padx=(5, 0))

        # Botones de control
        button_frame = ttk.Frame(control_frame, style="Panel.TFrame")
        button_frame.pack(side=tk.RIGHT)

        self.detection_var = tk.BooleanVar(value=True)
        detection_check = ttk.Checkbutton(button_frame, text="Detección Activa",
                                          variable=self.detection_var,
                                          command=self.toggle_detection)
        detection_check.pack(side=tk.LEFT, padx=(0, 10))

        self.record_button = ttk.Button(button_frame, text="Grabar", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 5))

        self.screenshot_button = ttk.Button(button_frame, text="Capturar", command=self.take_screenshot)
        self.screenshot_button.pack(side=tk.LEFT)

        # Panel de visualización
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo (visualización)
        left_frame = ttk.Frame(display_frame, style="Panel.TFrame", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_panel = ttk.Label(left_frame, style="Panel.TLabel")
        self.video_panel.pack(fill=tk.BOTH, expand=True)

        # Panel derecho (información)
        right_frame = ttk.Frame(display_frame, style="Panel.TFrame", padding=10, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Información de FPS
        fps_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        fps_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(fps_frame, text="FPS:", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        self.fps_label = ttk.Label(fps_frame, text="0.0", style="Panel.TLabel", font=("Segoe UI", 12))
        self.fps_label.pack(side=tk.RIGHT)

        # Lista de detecciones
        ttk.Label(right_frame, text="Objetos Detectados:", style="Panel.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.detection_listbox = tk.Listbox(right_frame, bg=self.panel_color, fg=self.text_color,
                                            font=("Segoe UI", 10), height=15,
                                            selectbackground=self.accent_color)
        self.detection_listbox.pack(fill=tk.BOTH, expand=True)

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo para comenzar")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN,
                               anchor=tk.W, padding=(10, 5))
        status_bar.pack(fill=tk.X, pady=(15, 0))

    def load_model(self):
        """Carga el modelo YOLOv8 seleccionado"""
        try:
            model_name = self.model_size.get()
            self.status_var.set(f"Cargando modelo {model_name}...")
            self.root.update()

            # Cargar el modelo desde Ultralytics
            self.model = YOLO(f"{model_name}.pt")

            self.status_var.set(f"Modelo {model_name} cargado correctamente")
        except Exception as e:
            self.status_var.set(f"Error al cargar modelo: {str(e)}")

    def change_model(self, event=None):
        """Cambia el modelo YOLO utilizado"""
        self.load_model()

    def update_conf_threshold(self, value):
        """Actualiza el umbral de confianza"""
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.1f}")

    def toggle_detection(self):
        """Activa/desactiva la detección de objetos"""
        self.detection_enabled = self.detection_var.get()

    def start_webcam(self):
        """Inicia la captura desde la webcam"""
        self.stop_video()

        # Intentar abrir la cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Error: No se pudo acceder a la cámara web")
            return

        self.source_path = "Webcam"
        self.is_video_playing = True
        self.status_var.set("Webcam iniciada")

        # Iniciar el bucle de video
        self.update_video()

    def select_video(self):
        """Selecciona un archivo de video para reproducir"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ],
            title="Seleccionar archivo de video"
        )

        if not file_path:
            return

        self.stop_video()

        # Abrir el video
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_var.set(f"Error: No se pudo abrir el video {file_path}")
            return

        self.source_path = file_path
        self.is_video_playing = True
        self.status_var.set(f"Reproduciendo video: {os.path.basename(file_path)}")

        # Iniciar el bucle de video
        self.update_video()

    def select_image(self):
        """Selecciona una imagen para analizar"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ],
            title="Seleccionar imagen"
        )

        if not file_path:
            return

        self.stop_video()

        # Cargar la imagen
        img = cv2.imread(file_path)
        if img is None:
            self.status_var.set(f"Error: No se pudo cargar la imagen {file_path}")
            return

        self.source_path = file_path
        self.is_video_playing = False
        self.status_var.set(f"Imagen cargada: {os.path.basename(file_path)}")

        # Procesar la imagen
        self.process_frame(img)

    def toggle_recording(self):
        """Inicia/detiene la grabación"""
        if not self.is_video_playing:
            self.status_var.set("No hay video en reproducción para grabar")
            return

        if not self.recording:
            # Iniciar grabación
            output_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")],
                title="Guardar video como"
            )

            if not output_path:
                return

            # Obtener dimensiones del frame
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]

                # Configurar writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

                self.recording = True
                self.record_button.config(text="Detener")
                self.status_var.set(f"Grabando video en {output_path}")
            else:
                self.status_var.set("Error: No hay frame actual para grabar")

        else:
            # Detener grabación
            if self.output_video:
                self.output_video.release()
                self.output_video = None

            self.recording = False
            self.record_button.config(text="Grabar")
            self.status_var.set("Grabación finalizada")

    def take_screenshot(self):
        """Captura la imagen actual y la guarda"""
        if self.current_frame is None:
            self.status_var.set("No hay imagen para capturar")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
            title="Guardar imagen como"
        )

        if not output_path:
            return

        cv2.imwrite(output_path, self.current_frame)
        self.status_var.set(f"Imagen guardada en {output_path}")

    def stop_video(self):
        """Detiene la reproducción de video"""
        self.is_video_playing = False

        if self.recording:
            self.toggle_recording()

        if self.cap:
            self.cap.release()
            self.cap = None

    def update_video(self):
        """Actualiza el fotograma de video"""
        if not self.is_video_playing:
            return

        # Leer el siguiente fotograma
        ret, frame = self.cap.read()

        if not ret:
            # Si es fin del video, reiniciar o detener
            if self.source_path != "Webcam":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_video()
                    self.status_var.set("Fin del video")
                    return
            else:
                self.stop_video()
                self.status_var.set("Error en la cámara web")
                return

        # Procesar el fotograma en un hilo separado para no bloquear la UI
        if not self.is_processing:
            self.is_processing = True
            threading.Thread(target=self.process_frame_thread, args=(frame,)).start()

        # Programar la siguiente actualización
        self.root.after(15, self.update_video)

    def process_frame_thread(self, frame):
        """Procesa un fotograma en un hilo separado"""
        start_time = time.time()
        self.process_frame(frame.copy())
        end_time = time.time()

        # Calcular FPS
        fps = 1 / (end_time - start_time)
        self.fps_avg = 0.9 * self.fps_avg + 0.1 * fps  # Promedio móvil

        # Actualizar interfaz desde el hilo principal
        self.root.after(0, self.update_fps_display)

        self.is_processing = False

    def update_fps_display(self):
        """Actualiza la visualización de FPS"""
        self.fps_label.config(text=f"{self.fps_avg:.1f}")

    def process_frame(self, frame):
        """Procesa un fotograma para detección de objetos"""
        if frame is None:
            return

        # Guardar el frame actual
        self.current_frame = frame.copy()

        # Grabar si está activado
        if self.recording and self.output_video:
            self.output_video.write(frame)

        # Realizar detección si está activada
        if self.detection_enabled:
            # Realizar detección con YOLOv8
            results = self.model(frame, conf=self.conf_threshold)

            # Procesar resultados
            self.detection_results = []

            # Limpiar lista de detecciones desde el hilo principal
            self.root.after(0, self.detection_listbox.delete, 0, tk.END)

            # Dibujar resultados en el frame
            for result in results:
                boxes = result.boxes

                for i, box in enumerate(boxes):
                    # Obtener coordenadas, confianza y clase
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]

                    # Guardar resultados
                    self.detection_results.append({
                        "class": cls_name,
                        "confidence": conf,
                        "box": [x1, y1, x2 - x1, y2 - y1]
                    })

                    # Añadir a la lista desde el hilo principal
                    label = f"{cls_name}: {conf:.2f}"
                    self.root.after(0, self.add_to_detection_list, label)

                    # Dibujar rectángulo y texto
                    color = self.get_color(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Añadir etiqueta con fondo
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convertir a formato para mostrar en Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mostrar frame desde el hilo principal
        self.root.after(0, self.update_image_display, frame_rgb)

    def add_to_detection_list(self, label):
        """Añade una detección a la lista"""
        self.detection_listbox.insert(tk.END, label)

    def update_image_display(self, frame_rgb):
        """Actualiza la imagen mostrada"""
        img = Image.fromarray(frame_rgb)

        # Redimensionar para ajustar al panel
        panel_width = self.video_panel.winfo_width() or 800
        panel_height = self.video_panel.winfo_height() or 600
        img = self.resize_image(img, panel_width, panel_height)

        # Mostrar en el panel
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_panel.configure(image=img_tk)
        self.video_panel.image = img_tk

    def resize_image(self, img, target_width, target_height):
        """Redimensiona una imagen manteniendo su relación de aspecto"""
        width, height = img.size
        ratio = min(target_width / width, target_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)

    def get_color(self, class_id):
        """Genera un color consistente para una clase"""
        # Generar colores determinísticos basados en el ID de clase
        np.random.seed(class_id * 10)
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        np.random.seed(None)  # Resetear seed
        return color

    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.stop_video()
        self.root.destroy()


def main():
    # Crear ventana principal
    root = tk.Tk()
    app = ObjectDetectorApp(root)

    # Configurar evento de cierre
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Iniciar bucle principal
    root.mainloop()


if __name__ == "__main__":
    main()