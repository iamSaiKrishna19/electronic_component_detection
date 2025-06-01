import yolov5
import cv2
import numpy as np
import yaml
import tkinter as tk
from PIL import Image, ImageTk
import time
import sqlite3
import os
import pandas as pd
import subprocess
from tabulate import tabulate
from tkinter import ttk
from tkinter import simpledialog


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

# Create a GUI window
window = tk.Tk()
window.title("Camera Object Detection")

# Create a label for displaying the frames
image_label = tk.Label(window)
image_label.pack()

# Create a label for displaying the label counts
counts_label = tk.Label(window, font=("Helvetica", 12))
counts_label.pack()

# Create a label for displaying the timer text
timer_text_label = tk.Label(window, text="Timer:", font=("Helvetica", 20))
timer_text_label.pack()

# Create a label for displaying the timer
timer_label = tk.Label(window, font=("Helvetica", 30))
timer_label.pack()

# Open the video capture
cap = cv2.VideoCapture(1)

# Set the video capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])

label_counts = {}
is_label_counted = False
freeze_time = None
freeze_duration = 3  # 3 seconds freeze duration
data_SQL = []
update_SQL = []
INDEX = ["ARDUINO_NANO", "DHT11", "ULTRASONIC", "BREADBOARD", "LCD_16*2"]

def readSQL():
    if os.path.isfile("data.db"):
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM my_table')
        rd = cursor.fetchall()
        data = {}
        for classe in rd:
            data[classe[1]] = classe[2]
        return data
    else:
        return {}

def writeToSQL(data):
    global data_SQL
    t_data = []

    for component in data:
        component.insert(0, INDEX.index(component[0]))
        component = tuple(component)
        t_data.append(component)

    if readSQL() == label_counts:
        return data

    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS my_table (id INTEGER PRIMARY KEY,name TEXT,qty INTEGER)''')
    cursor.executemany('INSERT OR REPLACE INTO my_table VALUES (?, ?, ?)', t_data)
    conn.commit()

    cursor.execute('SELECT * FROM my_table')
    data = cursor.fetchall()
    conn.close()
    return data

def convert_to_excel():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        inp = simpledialog.askstring("Excel Sheet Name", "Enter the name for your Excel sheet:")

        excel_file = f"{inp}.xlsx"
        df.to_excel(excel_file, index=True)
        print(f"Table '{table_name}' exported to {excel_file} successfully.")

    conn.close()

    

    # Open the Excel file
    subprocess.Popen(["start", excel_file], shell=True)



def display_data():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM my_table')
    data = cursor.fetchall()
    conn.close()

    if len(data) > 0:
        headers = ['ID', 'Name', 'Quantity']
        table = tabulate(data, headers, tablefmt='grid')

        # Create a GUI window
        window = tk.Tk()
        window.title("SQLite Data")

        # Create a table widget
        table_widget = ttk.Treeview(window, columns=headers, show='headings')
        for header in headers:
            table_widget.heading(header, text=header)
        for row in data:
            table_widget.insert('', 'end', values=row)
        table_widget.pack()

        window.mainloop()
    else:
        print("No data found in the database.")



def update_frame():
    global freeze_time

    ret, frame = cap.read()
    if not ret:
        return

    # Perform inference on the frame
    results = model(frame)

    # Parse the results
    predictions = results.pred[0].detach().cpu().numpy()

    if predictions.shape[0] > 0:
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # Apply non-maximum suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), model.conf, model.iou)

        # Check if indices are not empty
        if len(indices) > 0:
            # Show detection bounding boxes on the frame
            for i in indices.flatten():
                box = boxes[i]
                score = scores[i]
                category = categories[i]

                x1, y1, x2, y2 = np.round(box).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_label = labels[int(category)]
                cv2.putText(frame, f'{class_label}: {score:.2f}', (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

                if class_label in label_counts:
                    label_counts[class_label] += 1
                    freeze_time = time.time()
                else:
                    label_counts[class_label] = 0

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image format
    image = Image.fromarray(frame_rgb)

    # Resize the image to fit the label
    image = image.resize((640, 480))

    # Convert the PIL Image to Tkinter Image
    tk_image = ImageTk.PhotoImage(image)

    # Update the label with the new image
    image_label.configure(image=tk_image)
    image_label.image = tk_image

    # Update the label counts
    counts_text = "Label Counts:\n"
    data = []
    for label, count in label_counts.items():
        counts_text += f"{label}: {count}\n"
        data.append([label, count])
    counts_label.configure(text=counts_text)
    writeToSQL(data)

    # Check if the timer should be displayed
    if freeze_time is not None:
        current_time = time.time()
        elapsed_time = current_time - freeze_time

        # Check if the freeze duration has passed
        if elapsed_time >= freeze_duration:
            freeze_time = None
        else:
            # Update the timer label
            remaining_time = int(freeze_duration - elapsed_time)
            timer_label.configure(text=str(remaining_time))
            timer_label.after(1000, update_frame) # Update the timer every second
            return

    # Update the frame every millisecond
    window.after(1, update_frame)

# Start updating the frame
label_counts = readSQL()
update_frame()

# Create a button to convert SQLite data to Excel
excel_button = tk.Button(window, text="Convert to Excel", command=convert_to_excel)
excel_button.pack()

# Create a button to display SQLite data
display_button = tk.Button(window, text="Display Data", command=display_data)
display_button.pack()

# Start the Tkinter event loop
window.mainloop()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
