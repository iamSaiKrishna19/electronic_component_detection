# first_postgres.py
# This script replicates the logic from first.py but uses PostgreSQL instead of SQLite.
# Make sure you have psycopg2 installed: pip install psycopg2

import yolov5
import cv2
import numpy as np
import yaml
import tkinter as tk
from PIL import Image, ImageTk
import time
import psycopg2
import os
import pandas as pd
import subprocess
from tabulate import tabulate
from tkinter import ttk
from tkinter import simpledialog
import tkinter.messagebox

# PostgreSQL DB connection parameters
DB_PARAMS = {
    'dbname': 'electronic',  # Replace with your actual database name
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': 5432
}

def create_connection():
    conn = psycopg2.connect(**DB_PARAMS)
    return conn

def readSQL():
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM my_table')
        rd = cursor.fetchall()
        data = {}
        for classe in rd:
            data[classe[1]] = classe[2]
        return data
    except Exception:
        return {}
    finally:
        if 'conn' in locals():
            conn.close()

def writeToSQL(data):
    global data_SQL, user_email
    t_data = []
    for component in data:
        # Remove id insertion: do not add INDEX.index(component[0])
        # Only use name, qty, email
        name = component[0]
        qty = component[1]
        t_data.append((name, qty, user_email))
    if readSQL() == label_counts:
        return data
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS my_table (
        id SERIAL PRIMARY KEY,
        name TEXT,
        qty INTEGER,
        email TEXT,
        UNIQUE(email, name)
    )''')
    for tup in t_data:
        # Insert without id, let SERIAL handle it
        cursor.execute('''
            INSERT INTO my_table (name, qty, email)
            VALUES (%s, %s, %s)
            ON CONFLICT (email, name)
            DO UPDATE SET qty = my_table.qty + EXCLUDED.qty
        ''', tup)
    conn.commit()
    cursor.execute('SELECT * FROM my_table')
    data = cursor.fetchall()
    conn.close()
    return data

def convert_to_excel():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        inp = simpledialog.askstring("Excel Sheet Name", "Enter the name for your Excel sheet:")
        excel_file = f"{inp}.xlsx"
        df.to_excel(excel_file, index=True)
        print(f"Table '{table_name}' exported to {excel_file} successfully.")
    conn.close()
    subprocess.Popen(["start", excel_file], shell=True)

def display_data():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM my_table')
    data = cursor.fetchall()
    conn.close()
    if len(data) > 0:
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = tabulate(data, headers, tablefmt='grid')
        window = tk.Tk()
        window.title("PostgreSQL Data")
        table_widget = ttk.Treeview(window, columns=headers, show='headings')
        for header in headers:
            table_widget.heading(header, text=header)
        for row in data:
            table_widget.insert('', 'end', values=row)
        table_widget.pack()
        window.mainloop()
    else:
        print("No data found in the database.")

def search_by_email():
    search_email = simpledialog.askstring("Search Email", "Enter the email to search:")
    if not search_email:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM my_table WHERE email=%s', (search_email,))
    data = cursor.fetchall()
    conn.close()
    if len(data) > 0:
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = tabulate(data, headers, tablefmt='grid')
        window_search = tk.Tk()
        window_search.title(f"Data for {search_email}")
        table_widget = ttk.Treeview(window_search, columns=headers, show='headings')
        for header in headers:
            table_widget.heading(header, text=header)
        for row in data:
            table_widget.insert('', 'end', values=row)
        table_widget.pack()
        window_search.mainloop()
    else:
        tk.messagebox.showinfo("No Data", f"No data found for email: {search_email}")

def show_entire_database():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM my_table')
    data = cursor.fetchall()
    conn.close()
    if len(data) > 0:
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = tabulate(data, headers, tablefmt='grid')
        window_all = tk.Tk()
        window_all.title("Entire Database")
        table_widget = ttk.Treeview(window_all, columns=headers, show='headings')
        for header in headers:
            table_widget.heading(header, text=header)
        for row in data:
            table_widget.insert('', 'end', values=row)
        table_widget.pack()
        window_all.mainloop()
    else:
        tk.messagebox.showinfo("No Data", "No data found in the database.")

def delete_by_email():
    del_email = simpledialog.askstring("Delete by Email", "Enter the user's email:")
    if not del_email:
        return
    del_component = simpledialog.askstring("Delete by Email", "Enter the component name to delete for this user:")
    if not del_component:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM my_table WHERE email=%s AND name=%s', (del_email, del_component))
    conn.commit()
    conn.close()
    tk.messagebox.showinfo("Delete", f"Entry for email '{del_email}' and component '{del_component}' deleted (if existed).")

def main_menu():
    menu_root = tk.Tk()
    menu_root.title("Select Action")
    menu_root.geometry("300x320")
    label = tk.Label(menu_root, text="Choose an action:", font=("Helvetica", 14))
    label.pack(pady=20)
    def do_new():
        menu_root.destroy()
        start_detection_session()
    def do_search_name():
        menu_root.destroy()
        search_by_name()
    def do_search_email():
        menu_root.destroy()
        search_by_email()
    def do_delete_email():
        menu_root.destroy()
        delete_by_email()
    def do_all():
        menu_root.destroy()
        show_entire_database()
    new_btn = tk.Button(menu_root, text="Create New Entry", command=do_new, width=25, height=2)
    new_btn.pack(pady=5)
    search_name_btn = tk.Button(menu_root, text="Search by Component Name", command=do_search_name, width=25, height=2)
    search_name_btn.pack(pady=5)
    search_email_btn = tk.Button(menu_root, text="Search by Email", command=do_search_email, width=25, height=2)
    search_email_btn.pack(pady=5)
    delete_email_btn = tk.Button(menu_root, text="Delete by Email and Component", command=do_delete_email, width=25, height=2)
    delete_email_btn.pack(pady=5)
    all_btn = tk.Button(menu_root, text="Show Entire Database", command=do_all, width=25, height=2)
    all_btn.pack(pady=5)
    menu_root.mainloop()

def start_detection_session():
    global user_email, window
    window_email = tk.Tk()
    window_email.withdraw()
    user_email = simpledialog.askstring("Email", "Enter your email:")
    window_email.destroy()
    if not user_email:
        return
    start_detection_gui()

def start_detection_gui():
    global window, label_counts, is_label_counted, freeze_time, freeze_duration, data_SQL, update_SQL, INDEX
    window = tk.Tk()
    window.title("Camera Object Detection")
    image_label = tk.Label(window)
    image_label.pack()
    counts_label = tk.Label(window, font=("Helvetica", 12))
    counts_label.pack()
    timer_text_label = tk.Label(window, text="Timer:", font=("Helvetica", 20))
    timer_text_label.pack()
    timer_label = tk.Label(window, font=("Helvetica", 30))
    timer_label.pack()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])
    label_counts = {}
    is_label_counted = False
    freeze_time = None
    freeze_duration = 3
    data_SQL = []
    update_SQL = []
    INDEX = ["ARDUINO_NANO", "DHT11", "ULTRASONIC", "BREADBOARD", "LCD_16*2"]
    def update_frame():
        nonlocal cap
        global freeze_time
        ret, frame = cap.read()
        if not ret:
            return
        results = model(frame)
        predictions = results.pred[0].detach().cpu().numpy()
        new_detection = False
        if predictions.shape[0] > 0:
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), model.conf, model.iou)
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    score = scores[i]
                    category = categories[i]
                    x1, y1, x2, y2 = np.round(box).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_label = labels[int(category)]
                    cv2.putText(frame, f'{class_label}: {score:.2f}', (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if class_label in label_counts:
                        label_counts[class_label] += 1
                        freeze_time = time.time()
                        new_detection = True
                    else:
                        label_counts[class_label] = 1
                        freeze_time = time.time()
                        new_detection = True
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((640, 480))
        tk_image = ImageTk.PhotoImage(image)
        image_label.configure(image=tk_image)
        image_label.image = tk_image
        counts_text = "Label Counts:\n"
        data = []
        for label, count in label_counts.items():
            counts_text += f"{label}: {count}\n"
            data.append([label, count])
        counts_label.configure(text=counts_text)
        if new_detection:
            writeToSQL(data)
        if freeze_time is not None:
            current_time = time.time()
            elapsed_time = current_time - freeze_time
            if elapsed_time >= freeze_duration:
                freeze_time = None
            else:
                remaining_time = int(freeze_duration - elapsed_time)
                timer_label.configure(text=str(remaining_time))
                timer_label.after(1000, update_frame)
                return
        window.after(1, update_frame)
    update_frame()
    excel_button = tk.Button(window, text="Convert to Excel", command=convert_to_excel)
    excel_button.pack()
    display_button = tk.Button(window, text="Display Data", command=display_data)
    display_button.pack()
    window.mainloop()
    cap.release()
    cv2.destroyAllWindows()

def search_by_name():
    search_name = simpledialog.askstring("Search Name", "Enter the component name to search:")
    if not search_name:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM my_table WHERE name=%s', (search_name,))
    data = cursor.fetchall()
    conn.close()
    if len(data) > 0:
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = tabulate(data, headers, tablefmt='grid')
        window_search = tk.Tk()
        window_search.title(f"Data for {search_name}")
        table_widget = ttk.Treeview(window_search, columns=headers, show='headings')
        for header in headers:
            table_widget.heading(header, text=header)
        for row in data:
            table_widget.insert('', 'end', values=row)
        table_widget.pack()
        window_search.mainloop()
    else:
        tk.messagebox.showinfo("No Data", f"No data found for component: {search_name}")

# Load the YOLOv5 model
model = yolov5.load('best.pt')
# Set model parameters
model.conf = 0.80 # NMS confidence threshold
model.iou = 0.95 # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
# Set the desired input size
input_size = (640, 480)  # Replace with your desired size
# Load the class labels from YAML file
with open('ml.yaml', 'r', encoding='utf-8') as f:
    labels = yaml.safe_load(f)['names']

if __name__ == "__main__":
    main_menu()
