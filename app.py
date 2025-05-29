import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading


class ImageProcessor:
    @staticmethod
    def apply_contraharmonic_mean_filter(image, kernel_size=3, Q=-1.5, sigma=1.0, progress_callback=None):
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным")

        padding = kernel_size // 2

        if len(image.shape) == 3:
            channels = cv2.split(image)
            filtered_channels = []
            for i, ch in enumerate(channels):
                if progress_callback:
                    progress_callback(i * 100 / len(channels))
                filtered_channel = ImageProcessor.apply_contraharmonic_mean_filter(
                    ch, kernel_size, Q, sigma, None)
                filtered_channels.append(filtered_channel)
                if progress_callback:
                    progress_callback((i + 1) * 100 / len(channels))
            return cv2.merge(filtered_channels)

        padded_image = np.pad(image, padding, mode='constant', constant_values=0).astype(np.float32)

        x, y = np.indices((kernel_size, kernel_size))
        center = kernel_size // 2
        exponent = -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
        weights = np.exp(exponent)
        weights /= np.sum(weights)

        output = np.zeros_like(image, dtype=np.float32)
        total_pixels = image.shape[0] * image.shape[1]
        processed_pixels = 0

        for y_idx in range(padding, padded_image.shape[0] - padding):
            for x_idx in range(padding, padded_image.shape[1] - padding):
                region = padded_image[y_idx - padding:y_idx + padding + 1, x_idx - padding:x_idx + padding + 1]

                if Q < 0:
                    region = np.where(region == 0, 1e-6, region)

                num = np.sum(weights * (region ** (Q + 1)))
                denom = np.sum(weights * (region ** Q))

                if denom != 0:
                    output[y_idx - padding, x_idx - padding] = num / denom
                else:
                    output[y_idx - padding, x_idx - padding] = region[padding, padding]

                processed_pixels += 1
                if progress_callback and processed_pixels % 100 == 0:
                    progress = processed_pixels / total_pixels * 100
                    progress_callback(progress)

        if progress_callback:
            progress_callback(100)

        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_rayleigh_denoising_filter(image, noise_scale=25, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(10)

            if len(image.shape) == 3:
                channels = cv2.split(image)
                filtered_channels = []

                for i, channel in enumerate(channels):
                    if progress_callback:
                        progress_callback(10 + i * 30)

                    rayleigh_noise = np.random.rayleigh(noise_scale, channel.shape).astype(np.uint8)
                    noisy_channel = cv2.add(channel, rayleigh_noise)
                    noisy_channel = np.clip(noisy_channel, 0, 255).astype(np.uint8)

                    if progress_callback:
                        progress_callback(20 + i * 30)

                    denoised_channel = cv2.medianBlur(noisy_channel, 3)

                    if progress_callback:
                        progress_callback(25 + i * 30)

                    denoised_channel = cv2.fastNlMeansDenoising(denoised_channel, h=10,
                                                                templateWindowSize=7,
                                                                searchWindowSize=21)

                    filtered_channels.append(denoised_channel)

                    if progress_callback:
                        progress_callback(30 + i * 30)

                result = cv2.merge(filtered_channels)
            else:
                if progress_callback:
                    progress_callback(20)

                rayleigh_noise = np.random.rayleigh(noise_scale, image.shape).astype(np.uint8)
                noisy_image = cv2.add(image, rayleigh_noise)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

                if progress_callback:
                    progress_callback(50)

                denoised_image = cv2.medianBlur(noisy_image, 3)

                if progress_callback:
                    progress_callback(75)

                result = cv2.fastNlMeansDenoising(denoised_image, h=10,
                                                  templateWindowSize=7,
                                                  searchWindowSize=21)

            if progress_callback:
                progress_callback(100)

            return result

        except Exception as e:
            raise Exception(f"Ошибка при применении фильтра Рэлея: {str(e)}")


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool - Дипломный проект")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        self.original_image = None
        self.processed_image = None
        self.filtered_image = None
        self.filename = ""
        self.orig_img_tk = None
        self.proc_img_tk = None
        self.kernel_size = tk.IntVar(value=3)
        self.Q_value = tk.DoubleVar(value=-1.5)
        self.wb_r = tk.DoubleVar(value=1.0)
        self.wb_g = tk.DoubleVar(value=1.0)
        self.wb_b = tk.DoubleVar(value=1.0)
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.progress_var = tk.DoubleVar(value=0)
        self.current_task = None
        self.stop_processing = False

        self.create_widgets()
        self.setup_bindings()

    def create_widgets(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Image.TFrame', background='white')

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="Image Processing Tool", style='Header.TLabel').pack(side=tk.LEFT)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = ttk.Label(progress_frame, text="", anchor=tk.CENTER)
        self.progress_label.pack(fill=tk.X, pady=(5, 0))

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.create_image_frame(image_frame, "Оригинальное изображение", True)
        self.create_image_frame(image_frame, "Обработанное изображение", False)

        control_outer_frame = ttk.Frame(content_frame, width=300)
        control_outer_frame.pack(side=tk.RIGHT, fill=tk.Y)

        control_canvas = tk.Canvas(control_outer_frame, width=300)
        control_scrollbar = ttk.Scrollbar(control_outer_frame, orient=tk.VERTICAL, command=control_canvas.yview)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        control_frame = ttk.Frame(control_canvas)
        control_frame_window = control_canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)

        load_frame = ttk.LabelFrame(control_frame, text="Загрузка изображения")
        load_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(load_frame, text="Открыть изображение", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(load_frame, text="Сохранить результат", command=self.save_image).pack(fill=tk.X, pady=5)

        filters_frame = ttk.LabelFrame(control_frame, text="Фильтры обработки")
        filters_frame.pack(fill=tk.X, pady=(0, 10))

        params_frame = ttk.Frame(filters_frame)
        params_frame.pack(fill=tk.X, pady=5)
        ttk.Label(params_frame, text="Размер ядра:").pack(side=tk.LEFT)
        ttk.Spinbox(params_frame, from_=3, to=15, increment=2, textvariable=self.kernel_size, width=5).pack(
            side=tk.LEFT, padx=5)

        ttk.Button(filters_frame, text="Медианный фильтр", command=self.apply_median_filter).pack(fill=tk.X, pady=5)
        ttk.Button(filters_frame, text="Фильтр среднего арифметического", command=self.apply_mean_filter).pack(
            fill=tk.X, pady=5)
        ttk.Button(filters_frame, text="Фильтр среднего геометрического",
                   command=self.apply_geometric_mean_filter).pack(fill=tk.X, pady=5)

        contraharmonic_frame = ttk.Frame(filters_frame)
        contraharmonic_frame.pack(fill=tk.X, pady=5)
        ttk.Label(contraharmonic_frame, text="Q (порядок):").pack(side=tk.LEFT)
        ttk.Spinbox(contraharmonic_frame, from_=-5.0, to=5.0, increment=0.1, textvariable=self.Q_value, width=6).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(filters_frame, text="Контргармонический фильтр", command=self.apply_contraharmonic_filter).pack(
            fill=tk.X, pady=5)

        ttk.Button(filters_frame, text="Фильтр шума Рэлея", command=self.apply_rayleigh_filter).pack(
            fill=tk.X, pady=5)

        bc_frame = ttk.LabelFrame(control_frame, text="Яркость и контрастность")
        bc_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(bc_frame, text="Яркость:").pack(anchor=tk.W)
        ttk.Scale(bc_frame, from_=-100, to=100, variable=self.brightness, orient=tk.HORIZONTAL,
                  command=lambda e: self.adjust_brightness_contrast()).pack(fill=tk.X, pady=2)
        ttk.Label(bc_frame, text="Контрастность:").pack(anchor=tk.W)
        ttk.Scale(bc_frame, from_=0.1, to=3.0, variable=self.contrast, orient=tk.HORIZONTAL,
                  command=lambda e: self.adjust_brightness_contrast()).pack(fill=tk.X, pady=2)
        ttk.Button(bc_frame, text="Сбросить яркость/контраст", command=self.reset_brightness_contrast).pack(fill=tk.X,
                                                                                                            pady=5)
        wb_frame = ttk.LabelFrame(control_frame, text="Баланс белого")
        wb_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(wb_frame, text="Красный:").pack(anchor=tk.W)
        ttk.Scale(wb_frame, from_=0.5, to=2.0, variable=self.wb_r, orient=tk.HORIZONTAL,
                  command=lambda e: self.display_processed_image()).pack(fill=tk.X, pady=2)
        ttk.Label(wb_frame, text="Зеленый:").pack(anchor=tk.W)
        ttk.Scale(wb_frame, from_=0.5, to=2.0, variable=self.wb_g, orient=tk.HORIZONTAL,
                  command=lambda e: self.display_processed_image()).pack(fill=tk.X, pady=2)
        ttk.Label(wb_frame, text="Синий:").pack(anchor=tk.W)
        ttk.Scale(wb_frame, from_=0.5, to=2.0, variable=self.wb_b, orient=tk.HORIZONTAL,
                  command=lambda e: self.display_processed_image()).pack(fill=tk.X, pady=2)
        ttk.Button(wb_frame, text="Сбросить баланс белого", command=self.reset_white_balance_button).pack(fill=tk.X,
                                                                                                          pady=5)
        info_frame = ttk.LabelFrame(control_frame, text="Информация")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.info_label = ttk.Label(info_frame, text="Откройте изображение для начала работы")
        self.info_label.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Готов")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

        control_canvas.bind("<Configure>", lambda e: control_canvas.itemconfig(control_frame_window, width=e.width))
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.bind_all("<MouseWheel>",
                                lambda e: control_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def create_image_frame(self, parent, title, is_original):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10) if is_original else 0)

        canvas = tk.Canvas(frame, bg='white', highlightthickness=0)
        hscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        vscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)

        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        img_container = ttk.Frame(canvas, style='Image.TFrame', padding=0)
        img_id = canvas.create_window((0, 0), window=img_container, anchor=tk.NW)

        img_label = ttk.Label(img_container, background='white')
        img_label.pack()

        if is_original:
            self.orig_canvas = canvas
            self.orig_img_container = img_container
            self.orig_img_id = img_id
            self.orig_img_label = img_label
        else:
            self.proc_canvas = canvas
            self.proc_img_container = img_container
            self.proc_img_id = img_id
            self.proc_img_label = img_label

    def setup_bindings(self):
        self.orig_img_container.bind('<Configure>',
                                     lambda e: self.orig_canvas.configure(scrollregion=self.orig_canvas.bbox("all")))
        self.proc_img_container.bind('<Configure>',
                                     lambda e: self.proc_canvas.configure(scrollregion=self.proc_canvas.bbox("all")))
        self.orig_canvas.bind('<Configure>', lambda e: self.orig_canvas.itemconfig(self.orig_img_id, width=e.width))
        self.proc_canvas.bind('<Configure>', lambda e: self.proc_canvas.itemconfig(self.proc_img_id, width=e.width))
        self.orig_canvas.bind("<MouseWheel>", lambda e: self.orig_canvas.yview_scroll(-1 * int(e.delta / 120), "units"))
        self.proc_canvas.bind("<MouseWheel>", lambda e: self.proc_canvas.yview_scroll(-1 * int(e.delta / 120), "units"))

    def load_image(self):
        filetypes = (('Изображения', '*.jpg *.jpeg *.png *.bmp'), ('Все файлы', '*.*'))
        filename = filedialog.askopenfilename(title='Открыть изображение', filetypes=filetypes)
        if filename:
            self.filename = filename
            img = cv2.imread(filename)
            if img is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
                return

            self.original_image = img
            self.filtered_image = img.copy()
            self.processed_image = img.copy()
            self.display_original_image()
            self.display_processed_image()
            self.info_label.config(text=f"Загружено: {os.path.basename(filename)}")
            self.status_var.set("Изображение загружено")
            self.reset_white_balance_button()
            self.reset_brightness_contrast()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Внимание", "Нет обработанного изображения для сохранения.")
            return

        filetypes = (('PNG', '*.png'), ('JPEG', '*.jpg;*.jpeg'), ('BMP', '*.bmp'))
        save_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=filetypes)
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            self.status_var.set(f"Изображение сохранено: {os.path.basename(save_path)}")

    def display_original_image(self):
        if self.original_image is None:
            return
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.orig_img_tk = ImageTk.PhotoImage(pil_img)
        self.orig_img_label.config(image=self.orig_img_tk)

    def display_processed_image(self):
        if self.processed_image is None:
            return

        img = self.processed_image.copy()
        if len(img.shape) == 3:
            b, g, r = cv2.split(img.astype(np.float32))
            r *= self.wb_r.get()
            g *= self.wb_g.get()
            b *= self.wb_b.get()
            img = cv2.merge([b, g, r])
            img = np.clip(img, 0, 255).astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.proc_img_tk = ImageTk.PhotoImage(pil_img)
        self.proc_img_label.config(image=self.proc_img_tk)

    def apply_median_filter(self):
        if not self.check_image_loaded():
            return

        k = self.kernel_size.get()
        if k % 2 == 0:
            messagebox.showerror("Ошибка", "Размер ядра должен быть нечетным числом.")
            return

        self.start_processing("Применение медианного фильтра...")

        def process():
            try:
                filtered = cv2.medianBlur(self.original_image, k)
                self.update_filtered_image(filtered, "Применён медианный фильтр")
            except Exception as e:
                self.handle_processing_error(e)
            finally:
                self.complete_processing()

        self.run_in_thread(process)

    def apply_mean_filter(self):
        if not self.check_image_loaded():
            return

        k = self.kernel_size.get()
        if k % 2 == 0:
            messagebox.showerror("Ошибка", "Размер ядра должен быть нечетным числом.")
            return

        self.start_processing("Применение фильтра среднего арифметического...")

        def process():
            try:
                kernel = np.ones((k, k), np.float32) / (k * k)
                filtered = cv2.filter2D(self.original_image, -1, kernel)
                self.update_filtered_image(filtered, "Применён фильтр среднего арифметического")
            except Exception as e:
                self.handle_processing_error(e)
            finally:
                self.complete_processing()

        self.run_in_thread(process)

    def apply_geometric_mean_filter(self):
        if not self.check_image_loaded():
            return

        k = self.kernel_size.get()
        if k % 2 == 0:
            messagebox.showerror("Ошибка", "Размер ядра должен быть нечетным числом.")
            return

        self.start_processing("Применение фильтра среднего геометрического...")

        def process():
            try:
                img = self.original_image.astype(np.float32) + 1e-6

                def process_channel(channel):
                    height, width = channel.shape
                    padded = np.pad(channel, k // 2, mode='edge')
                    result = np.zeros_like(channel)

                    for y in range(height):
                        for x in range(width):
                            region = padded[y:y + k, x:x + k]
                            log_region = np.log(region)
                            result[y, x] = np.exp(log_region.mean())
                        self.update_progress(y / height * 100)
                        if self.stop_processing:
                            return None
                    return result

                if len(img.shape) == 3:
                    channels = cv2.split(img)
                    filtered_channels = []
                    for i, ch in enumerate(channels):
                        filtered_ch = process_channel(ch)
                        if filtered_ch is None:  # Processing was stopped
                            return
                        filtered_channels.append(filtered_ch)
                        self.update_progress((i + 1) * 100 / len(channels))
                    filtered = cv2.merge(filtered_channels)
                else:
                    filtered = process_channel(img)
                    if filtered is None:  # Processing was stopped
                        return

                filtered = np.clip(filtered, 0, 255).astype(np.uint8)
                self.update_filtered_image(filtered, "Применён фильтр среднего геометрического")
            except Exception as e:
                self.handle_processing_error(e)
            finally:
                self.complete_processing()

        self.run_in_thread(process)

    def apply_contraharmonic_filter(self):
        if not self.check_image_loaded():
            return

        k = self.kernel_size.get()
        Q = self.Q_value.get()
        if k % 2 == 0:
            messagebox.showerror("Ошибка", "Размер ядра должен быть нечетным числом.")
            return

        self.start_processing("Применение контргармонического фильтра...")

        def process():
            try:
                def progress_callback(percent):
                    self.update_progress(percent)
                    return not self.stop_processing

                filtered = ImageProcessor.apply_contraharmonic_mean_filter(
                    self.original_image, kernel_size=k, Q=Q, sigma=1.0,
                    progress_callback=progress_callback)

                if not self.stop_processing:
                    self.update_filtered_image(filtered, f"Применён контргармонический фильтр (Q={Q:.2f})")
            except Exception as e:
                self.handle_processing_error(e)
            finally:
                self.complete_processing()

        self.run_in_thread(process)

    def apply_rayleigh_filter(self):
        if not self.check_image_loaded():
            return

        self.start_processing("Фильтрация зашумленного изображения (Рэлея)...")

        def process():
            try:
                # Разделяем цветное изображение на каналы
                channels = cv2.split(self.original_image)

                denoised_channels = []
                for ch in channels:
                    med = cv2.medianBlur(ch, 3)
                    denoised = cv2.fastNlMeansDenoising(med, h=10, templateWindowSize=7, searchWindowSize=21)
                    denoised_channels.append(denoised)

                result = cv2.merge(denoised_channels)

                self.update_filtered_image(result, "Фильтр Рэлея применён")
            except Exception as e:
                self.handle_processing_error(e)
            finally:
                self.complete_processing()

        self.run_in_thread(process)

    def check_image_loaded(self):
        if self.original_image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение.")
            return False
        return True

    def start_processing(self, message):
        self.stop_processing = False
        self.progress_var.set(0)
        self.progress_label.config(text=message)
        self.root.update_idletasks()

    def update_progress(self, percent):
        self.progress_var.set(percent)
        if percent % 5 == 0:
            self.root.update_idletasks()

    def update_filtered_image(self, filtered_image, status_message):
        self.filtered_image = filtered_image
        self.apply_brightness_contrast_to_filtered()
        self.status_var.set(status_message)

    def handle_processing_error(self, error):
        print(f"Ошибка обработки: {error}")
        self.status_var.set("Ошибка обработки изображения")
        messagebox.showerror("Ошибка", f"Произошла ошибка при обработке: {str(error)}")

    def complete_processing(self):
        if not self.stop_processing:
            self.progress_var.set(100)
        self.progress_label.config(text="Обработка завершена" if not self.stop_processing else "Обработка прервана")
        self.display_processed_image()
        self.root.after(2000, lambda: self.progress_label.config(text=""))

    def run_in_thread(self, target):
        if self.current_task and self.current_task.is_alive():
            self.stop_processing = True
            self.current_task.join(timeout=0.1)

        self.current_task = threading.Thread(target=target)
        self.current_task.daemon = True
        self.current_task.start()

    def reset_white_balance_button(self):
        self.wb_r.set(1.0)
        self.wb_g.set(1.0)
        self.wb_b.set(1.0)
        self.display_processed_image()

    def adjust_brightness_contrast(self):
        if self.filtered_image is None:
            return
        self.apply_brightness_contrast_to_filtered()
        self.display_processed_image()
        self.status_var.set(f"Яркость: {self.brightness.get()}, Контрастность: {self.contrast.get():.2f}")

    def apply_brightness_contrast_to_filtered(self):
        if self.filtered_image is None:
            return

        brightness = self.brightness.get()
        contrast = self.contrast.get()
        img = self.filtered_image.copy().astype(np.float32)
        img = img * contrast + brightness
        self.processed_image = np.clip(img, 0, 255).astype(np.uint8)

    def reset_brightness_contrast(self):
        self.brightness.set(0)
        self.contrast.set(1.0)
        if self.filtered_image is not None:
            self.processed_image = self.filtered_image.copy()
            self.display_processed_image()
        self.status_var.set("Яркость и контрастность сброшены")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
