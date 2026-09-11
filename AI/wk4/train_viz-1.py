import sys
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,  QShortcut, QLabel, QCheckBox, QGridLayout, QGroupBox, QFormLayout, QSizePolicy, QFrame, QLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QKeySequence, QPalette, QImage, QPixmap
from PyQt5.QtCore import Qt, QRect, QTimer, QSize
import sklearn




class ModelManager:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(8, 8, 1)),
            tf.keras.layers.Conv2D(filters=4, activation='relu', kernel_size=(3, 3), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = 20, activation='relu'),
            tf.keras.layers.Dense(units = 10, activation='softmax')
        ])

        self.activation_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=[layer.output for layer in self.model.layers])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.accuracy = None
        self.gradients = None

        supervised_training_data = sklearn.datasets.load_digits()
        image_data = supervised_training_data.images
        correct_answers = supervised_training_data.target
        image_data = image_data / 16.0
        all_questions = image_data.reshape((-1, 8, 8, 1))
        all_answers = tf.keras.utils.to_categorical(correct_answers, num_classes=10)

        training_questions, self.test_questions, training_answers, self.test_answers = sklearn.model_selection.train_test_split(all_questions, all_answers, test_size=0.2)

        self.batches = list(tf.data.Dataset.from_tensor_slices((training_questions, training_answers)).batch(16, drop_remainder=True))
        self.epoch = 0
        self.batch_number = 0
        self.predictions = None


    def evaluate_accuracy(self):
        self.accuracy = self.model.evaluate(self.test_questions, self.test_answers, verbose=0)[1]  

    def get_activations(self, input_data):
        return self.activation_model.predict(input_data, verbose=0)      
   
    def train_step(self, questions, answers):
        with tf.GradientTape() as tape:
            self.predictions = self.model(questions, training=True)
            self.loss = self.loss_fn(answers, self.predictions)
        self.gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.model.trainable_variables))   # most time here


    # -------------------------------------------------------------------------
    # Main Window 

    def createMainWindow(self, window):
        self.canvas = DrawingCanvas(model)
        self.visualization = VisualizationPanel(model)

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.visualization)
        layout.addLayout(left_layout)
        layout.addWidget(scroll_area)
        window.setLayout(layout)
        window.setMinimumSize(1280, 720)
        window.setWindowTitle("CNN Visualization")
        window.showMaximized()  

    def onCloseMainWindow(self, main_window):
        self.canvas.close()
        self.visualization.close()

    # -------------------------------------------------------------------------
    # DrawingCanvas Window 

    def run_stop(self):
        if self.running:
            self.run_button.setText("Run")
            self.next_button.setEnabled(True)
            self.running = False
        else:
            self.run_button.setText("Stop")
            self.next_button.setEnabled(False)
            self.running = True
            self.run_iteration()

    def run_iteration(self):
        if self.running:       
            self.next_batch()
            QTimer.singleShot(0, self.run_iteration)

    def get_correct_answer(self, x, y):
        if self.all_correct_answers is not None:
            for i in range(10):
                if self.all_correct_answers[4*x+y, i] == 1:
                    return i
        return None
     
    def get_predicted_answer(self, x, y):
        if self.predictions is not None:
            best_digit = None
            best_score = 0
            for i in range(10):
                score = self.predictions[4*x+y, i]
                if score > best_score:
                    best_score = score
                    best_digit = i
            return best_digit
        return None            
    

    def make_cell_placeholder(self, text: str, is_digit: bool = False):
        # Outer frame: fixed total size, shows the border
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)  # styling via stylesheet
        frame.setFixedSize(QSize(self.CELL_OUTER, self.CELL_OUTER))
        frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        frame.setStyleSheet(f"""
            QFrame {{
                border: {self.BORDER_W}px solid #0066CC;
                border-radius: 0px;
                background: #FFFFFF;
            }}
        """)

        # Inner label: fixed 64×64 content area
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(QSize(self.CELL_SIZE, self.CELL_SIZE))
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lbl.setFont(QFont("Arial", 36 if is_digit else 10))
        #lbl.setStyleSheet("QLabel { color: #333; background: transparent; }")

        # Centre the label inside the frame
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addStretch()
        lay.addWidget(lbl, alignment=Qt.AlignCenter)
        lay.addStretch()

        return frame, lbl


    def CreateDrawingCanvas(self, window):
        self.window = window
        root = QVBoxLayout(window)
        window.setLayout(root)

        self.images = None
        self.running = False
        self.selected = None
        self.loss = None
        self.accuracy = None

        self.pal_green = QPalette()
        self.pal_green.setColor(QPalette.WindowText, QColor(0, 128, 0))  # dark green

        self.pal_red = QPalette()
        self.pal_red.setColor(QPalette.WindowText, QColor(255, 0, 0))    # red
        
        self.CELL_SIZE = 64        # content area (label/pixmap) size
        self.BORDER_W  = 1         # border width on the outer frame
        self.CELL_OUTER = self.CELL_SIZE + 2 * self.BORDER_W  # total frame size

        window.setFixedWidth(300)

        self.question_frames = [[None] * 4 for _ in range(4)]   # border containers
        self.question_labels = [[None] * 4 for _ in range(4)]   # 64x64 content labels
        self.answer_frames   = [[None] * 4 for _ in range(4)]
        self.answer_labels   = [[None] * 4 for _ in range(4)]

    
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(20)

        q_grid = QGridLayout()
        q_grid.setHorizontalSpacing(0)
        q_grid.setVerticalSpacing(0)
        q_grid.setSizeConstraint(QLayout.SetFixedSize)

        q_group = QGroupBox("Input Images")
        q_group.setLayout(q_grid)
        q_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        root.addWidget(q_group)

        self.frame_index = {}

        for i in range(4):
            for j in range(4):
                frame, lbl = self.make_cell_placeholder(f"Img")
                self.question_frames[i][j] = frame
                self.question_labels[i][j] = lbl
                q_grid.addWidget(frame, j, i)
                self.frame_index[frame] = (j, i)
        
        a_grid = QGridLayout()
        a_grid.setHorizontalSpacing(0)
        a_grid.setVerticalSpacing(0)
        a_grid.setSizeConstraint(QLayout.SetFixedSize)

        a_group = QGroupBox("Correct Answers")
        a_group.setLayout(a_grid)
        a_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        root.addWidget(a_group)

        for i in range(4):
            for j in range(4):
                frame, lbl = self.make_cell_placeholder("", is_digit=True)
                self.answer_frames[i][j] = frame
                self.answer_labels[i][j] = lbl
                a_grid.addWidget(frame, j, i)
                self.frame_index[frame] = (j, i)

        controls = QHBoxLayout()
        self.next_button = QPushButton("Next Batch")
        self.next_button.clicked.connect(lambda: self.next_batch())
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(lambda: self.run_stop())
        controls.addStretch()
        controls.addWidget(self.next_button)
        controls.addStretch()
        controls.addWidget(self.run_button)  
        controls.addStretch()
        root.addLayout(controls)

        # --- Metrics (placeholders only) ---
        metrics_group = QGroupBox("Training Metrics")
        metrics_form = QFormLayout()
        metrics_group.setLayout(metrics_form)

        self.lbl_loss = QLabel('-')
        self.lbl_epoch = QLabel('-')
        self.lbl_batch = QLabel('-')
        self.lbl_accuracy = QLabel('-')

        metrics_form.addRow("Average Loss:", self.lbl_loss)
        metrics_form.addRow("Epoch:",        self.lbl_epoch)
        metrics_form.addRow("Batch:",        self.lbl_batch)
        metrics_form.addRow("Accuracy:",     self.lbl_accuracy)
        root.addWidget(metrics_group)
        self.update_metrics()

    def image8x8_to_pixmap(self, img8x8):
        qimg = QImage(8, 8, QImage.Format_RGB32)
        for y in range(8):
            for x in range(8):
                g = int(img8x8[y][x] * 255)
                qimg.setPixelColor(x, y, QColor(g, g, g))
        return QPixmap.fromImage(qimg)

    def update_canvas(self):
        for x in range(4):
            for y in range(4):
                image = self.images[x][y]
                pixmap = self.image8x8_to_pixmap(image)
                label = self.question_labels[x][y]
                label.setPixmap(pixmap)
                label.setScaledContents(True) 
                label.setFixedSize(64,64)

        for y in range(4):
            for x in range(4):
                correct = self.get_correct_answer(x, y)
                predicted = self.get_predicted_answer(x, y)
                label = self.answer_labels[x][y]
                label.setText(str(correct))
                is_correct = (correct == predicted)
                pal = self.pal_green if is_correct else self.pal_red
                if label.palette() is not pal:
                    label.setPalette(pal)

        self.update_metrics()

    def update_metrics(self):
        self.lbl_loss.setText(f"{self.loss:.3f}" if self.loss is not None else "—")
        self.lbl_epoch.setText(str(self.epoch))
        self.lbl_batch.setText(str(self.batch_number) if self.batch_number is not None else "—")
        self.lbl_accuracy.setText(f"{100.0 * self.accuracy:.2f}%" if self.accuracy is not None else "—")
   
      
    def onCanvasMousePress(self, canvas_window, event):
        if event.button() != Qt.LeftButton:
            return
            
        self.selected = None

        pos = canvas_window.mapFromGlobal(event.globalPos())
        label = canvas_window.childAt(pos)
        if isinstance(label, QLabel):
            frame = label.parentWidget()
            if isinstance(frame, QFrame):
                y, x = self.frame_index[frame]
                self.selected = (x,y)
                self.on_new_image(self.images[x,y], 4*x+y)     
                canvas_window.update()


    # -------------------------------------------------------------------------
    # VisualizationPanel Window 

    def CreateVisualizationPanel(self, window):
        self.activations = []
        self.input_data = None
        self.get_loss = None

    def onVisualizationResize(self, window, event):
        self.scale_x = scale_x = window.width() / 1000
        self.scale_y = scale_y = window.height() / 700
        self.scale = scale = min(scale_x, scale_y)   
        self.radius = int(8 *scale)
        margin = 40 * scale
        spacing = 18 * scale
        for fnt_size in range(20, 1, -1):
            self.neuron_font = QFont("Arial", int(fnt_size * self.scale))
            self.metrics = QFontMetrics(self.neuron_font)
            if self.metrics.horizontalAdvance("0.0") < self.radius * 1.5 and self.metrics.height() < self.radius * 1.5:
                break
        self.edges = []
        self.neurons = []
        current_x = margin
        self.input = []
        for x in range(8):
            self.input.append([])
            for y in range(8):     
                neuron = Neuron(current_x + x * spacing, (238 * scale_y) + margin + y * spacing, 'input', x=x, y=y)
                self.neurons.append(neuron)
                self.input[x].append(neuron)  
        current_x += 8 * spacing + margin 
        self.conv = []
        for f in range(4):
            self.conv.append([])
            for x in range(8):
                self.conv[f].append([])
                for y in range(8):         
                    neuron = Neuron(current_x + x * spacing, margin + (y + f * 9) * spacing, 'conv', f=f, x=x, y=y)
                    self.neurons.append(neuron)
                    self.conv[f][x].append(neuron)
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            iy = y + dy
                            ix = x + dx
                            if 0 <= iy < 8 and 0 <= ix < 8:
                                self.edges.append(Edge(self.input[ix][iy], neuron))              
        current_x += 8 * spacing + margin + (50 * scale_x)
        self.pool = []
        for f in range(4):
            self.pool.append([])
            for x in range(4):
                self.pool[f].append([])
                for y in range(4):    
                    neuron = Neuron(current_x + x * spacing, margin + (y + f * 9 + 2) * spacing, 'pool', f=f, x=x, y=y)
                    self.neurons.append(neuron)
                    self.pool[f][x].append(neuron)          
                    for dy in range(0, 2):
                        for dx in range(0, 2):
                            self.edges.append(Edge(self.conv[f][x * 2 + dx][y * 2 + dy], neuron))
        current_x += 4 * spacing + margin + (70 * scale_x) 
        self.dense = []
        for i in range(20):
           neuron = Neuron(current_x, margin + i * (30 * scale_y), 'dense', y=i)
           self.neurons.append(neuron)
           self.dense.append(neuron)
           for f in range(4):
                for x in range(4):  
                    for y in range(4):
                        self.edges.append(Edge(self.pool[f][x][y], neuron))   
        current_x += spacing + margin + (70 * scale_x)
        self.output = []
        for j in range(10):
            neuron = Neuron(current_x, margin + scale_y * (j * 60 + 20), 'output', y=j)
            self.neurons.append(neuron)
            self.output.append(neuron)
            for i in range(20):
                self.edges.append(Edge(self.dense[i], neuron))       

    def drawNode(self, painter, neuron, min_values, max_values):   
        min_value, max_value = neuron.get_bounds(min_values, max_values)
        if min_value == max_value:
            intensity = 127
        else:
            intensity = int(255 * (neuron.activation - min_value) / (max_value - min_value))

        painter.setBrush(QColor(intensity, intensity, intensity))
        (px,py) = neuron.pos          
        painter.setPen(Qt.white)
        painter.drawEllipse(px-self.radius, py-self.radius, self.radius*2, self.radius*2)
        text = f'{neuron.activation:.1f}'          
        painter.setFont(self.neuron_font)
        if intensity > 127:
            painter.setPen(Qt.black)
        else:
            painter.setPen(Qt.white)
        painter.drawText(px-self.metrics.horizontalAdvance(text)//2, py+self.metrics.height()//4, text)                  
    
    def get_gradient(self, src, dst):
        if hasattr(self, 'gradients') and self.gradients is not None:
            match (src.layer, dst.layer):
                case ('input', 'conv'):
                    if dst.x == 1 and dst.y == 1: 
                        dx = dst.x - src.x
                        dy = dst.y - src.y
                        if -1 <= dx <= 1 and -1 <= dy <= 1:
                            return self.gradients[0][dy + 1, dx + 1, 0, dst.f].numpy()
                    return 0
                case ('conv', 'pool'):
                    return 0
                case ('pool', 'dense'):
                    return self.gradients[2][(src.y * 4 * 4) + (src.x * 4) + src.f, dst.y].numpy()
                case ('dense', 'output'):
                    return self.gradients[4][src.y, dst.y].numpy()
        return None 
    
    def gradient_to_line_width(self, g):
        g = abs(g)
        if g < 1e-6:
            return 0
        elif g < 1e-5:
            return 0.025
        elif g < 1e-4:
            return 0.05
        elif g < 1e-3:
            return 0.1
        elif g < 1e-2:
            return 0.2
        elif g < 1e-1:
            return 0.4
        elif g < 1:
            return 0.8
        else:
            return g
    
    def onPaintVisualization(self, window, event):    
        painter = QPainter(window)
        painter.setRenderHint(QPainter.Antialiasing)   
        painter.fillRect(window.rect(), QColor(100,100,0))
        painter.setFont(QFont("Arial", 10))

        # paint blue lines first
        for edge in self.edges:
            gradient = self.get_gradient(edge.src, edge.dst)
            if gradient is not None and gradient != 0:
                width = self.gradient_to_line_width(gradient)
                if gradient > 0:
                    painter.setPen(QPen(Qt.blue, width))
                elif gradient < 0:
                    painter.setPen(QPen(Qt.red, width))
                painter.drawLine(edge.src.pos[0], edge.src.pos[1], edge.dst.pos[0], edge.dst.pos[1])

        # then paint the nodes on top of the lines 
        min_values = [layer.min() for layer in self.activations]
        max_values = [layer.max() for layer in self.activations]
        for neuron in self.neurons:
            neuron.update_activation(self.input_data, self.activations)
            self.drawNode(painter, neuron, min_values, max_values)

        # add the labels next to each output category  
        painter.setFont(QFont("Arial", 20))
        for y in range(10):
            neuron = self.output[y]        
            painter.drawText(neuron.pos[0] + 20, neuron.pos[1] + 10, str(y))      

        # determine which output has highest weight
        best_digit = None
        best_score = -10000              
        for y in range(10):
            neuron = self.output[y]            
            if neuron.activation > best_score:
                best_digit = y
                best_score = neuron.activation
        painter.setFont(QFont("Arial", 15))   
        painter.drawText(self.input[0][0].pos[0], 20, "Input Layer")
        painter.drawText(self.conv[0][0][0].pos[0], 20, "Conv2D Layer")
        painter.drawText(self.pool[0][0][0].pos[0], 20, "MaxPooling Layer")
        painter.drawText(self.dense[0].pos[0], 20, "Dense Layer")  
        painter.drawText(self.output[0].pos[0], 20, "Output Layer") 

        if best_digit:   
            painter.setFont(QFont("Arial", 80))
            p4 = self.output[4].pos
            painter.drawText(int(p4[0] + self.scale_x * 250), int(p4[1]), str(best_digit))     

        painter.setFont(QFont("Arial", 12))
        for y in range(10):
            neuron = self.output[y]        
            painter.drawText(int(neuron.pos[0] + self.scale_x * 50), neuron.pos[1], f'{neuron.activation:.3f}')
            if hasattr(self, 'correct_answers'):
                painter.drawText(int(neuron.pos[0] + self.scale_x * 100), neuron.pos[1], f'{self.correct_answers[y]:.1f}')
                diff = self.correct_answers[y] - neuron.activation 
                if diff == 0:
                    painter.setPen(QColor(Qt.black))
                elif diff < 0:
                    painter.setPen(QColor(Qt.red))
                else:
                    painter.setPen(QColor(Qt.blue)) 
                painter.drawText(int(neuron.pos[0] + self.scale_x * 150), neuron.pos[1], f'{diff:+.3f}')
                painter.setPen(QColor(Qt.white))


    # -------------------------------------------------------------------------
    # Other Events (ToDo)


    def next_batch(self):

        if self.batch_number >= len(self.batches):

            # log weights at end of each epoch

            self.epoch += 1
            self.batch_number = 0

        if self.epoch == 0 or self.batch_number == 0:
            self.evaluate_accuracy()

        (self.batch_questions, self.batch_answers) = self.batches[self.batch_number]
        self.train_step(self.batch_questions, self.batch_answers)
        self.batch_number += 1

        self.images = self.batch_questions.numpy().reshape(4, 4, 8, 8)
        self.all_correct_answers = self.batch_answers.numpy()
        self.predictions = self.predictions.numpy()

        if self.selected:
            x,y = self.selected
            self.on_new_image(self.images[x,y], 4*x+y)
      
        self.update_canvas() 
        self.visualization.update()   
    
    def on_new_image(self, image, index):
        input_data = image.reshape(1, 8, 8, 1)
        self.activations = self.get_activations(input_data)
        self.index = index
        self.input_data = input_data
        self.correct_answers = self.batch_answers[index]
        self.visualization.update()

   

class Neuron:
    def __init__(self, pos_x, pos_y, layer, **coord):
        self.pos = (int(pos_x),int(pos_y))
        self.layer = layer
        self.edges = []
        self.highlighted = None        
        for key, value in coord.items():
            setattr(self, key, value)

    def update_activation(self, input_data, activations):
        if input_data is not None and activations is not None:
            match self.layer:
                case 'input':  self.activation = input_data[0, self.y, self.x, 0]
                case 'conv':   self.activation = activations[0][0, self.y, self.x, self.f]
                case 'pool':   self.activation = activations[1][0, self.y, self.x, self.f]
                case 'dense':  self.activation = activations[3][0, self.y]
                case 'output': self.activation = activations[4][0, self.y]
        else:
            self.activation = 0
                        
    def get_bounds(self, min_values, max_values):
        if min_values and max_values:
            match self.layer:
                case 'input':  return (0,1)
                case 'conv':   return min_values[0], max_values[0]
                case 'pool':   return min_values[1], max_values[1]
                case 'dense':  return min_values[3], max_values[3]
                case 'output': return (0,1)   
        else:
            return (0,1)
        
class Edge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

class MainWindow(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        model.createMainWindow(self)

    def closeEvent(self, event):
        self.model.onCloseMainWindow(self)
        super().closeEvent(event)     


class DrawingCanvas(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        model.CreateDrawingCanvas(self)
    
    def mousePressEvent(self, event):
        self.model.onCanvasMousePress(self, event)

class VisualizationPanel(QWidget):  
    def __init__(self, model):
        super().__init__()
        self.model = model
        model.CreateVisualizationPanel(self)

    def resizeEvent(self, event):
        self.model.onVisualizationResize(self, event)
    
    def paintEvent(self, event):    
        self.model.onPaintVisualization(self, event)                   


if __name__ == "__main__":
    model = ModelManager()
    app = QApplication(sys.argv)
    window = MainWindow(model)
    sys.exit(app.exec_())