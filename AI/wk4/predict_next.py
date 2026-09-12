# Import llama before PyQt5
from llama_cpp import Llama

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QTextEdit, QPushButton,
    QDoubleSpinBox, QLabel, QHeaderView, QAbstractItemView, QSizePolicy
)
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase

import sys, math, threading

# --- Model configuration ---
# model_name = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
model_name = "/home/yun/dev/uni/220/wk4/llama-3.2-3b-instruct-q4_k_m.gguf"
# NOTE: For maximum responsiveness you can set logits_all=False, but we keep your choice.
llm = Llama(model_path=model_name, logits_all=True, verbose=False)

TOP_K_DISPLAY = 100  # show 100 rows, request 100 logprobs


# ---------------- Helpers ----------------

def display_token_text(tok: str) -> str:
    """Single-quoted, escaped representation for display."""
    escaped = tok.encode("unicode_escape").decode("ascii")
    return f"'{escaped}'"

def token_codepoints_tooltip(tok: str) -> str:
    """Tooltip listing Unicode code points of the token."""
    if tok == "":
        return "(empty token)"
    return " ".join(f"U+{ord(ch):04X}" for ch in tok)

def token_id_repr(tok: str):
    """
    Best-effort mapping from token string to ID(s).
    Single-piece tokens -> one ID; multi-piece -> comma-separated; none -> '-'.
    """
    ids = llm.tokenize(tok.encode("utf-8"), add_bos=False)
    if not ids:
        return "-"
    return ",".join(map(str, ids))


# ---------------- Workers ----------------

class PreviewWorker(QThread):
    """One-shot preview for manual steps."""
    previewReady = pyqtSignal(str, list)  # (chosen_token, items)
    def __init__(self, preview_callable):
        super().__init__()
        self.preview_callable = preview_callable
    def run(self):
        tok, items = self.preview_callable()
        self.previewReady.emit(tok, items)


class AutoWorker(QThread):
    """Continuous preview loop for Continue mode."""
    tokenPreview = pyqtSignal(str, list)  # (chosen_token, items)
    def __init__(self, preview_callable):
        super().__init__()
        self.preview_callable = preview_callable
        self.running = True
        self.ack = threading.Event()
    def run(self):
        while self.running:
            tok, items = self.preview_callable()
            if not tok:
                break
            self.tokenPreview.emit(tok, items)
            self.ack.wait()
            self.ack.clear()
    def stop(self):
        self.running = False
        self.ack.set()


# ---------------- Main Widget ----------------

class TokenPicker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Next-token Picker")

        self.current_text = ""
        self.worker = None
        self.prev_worker = None
        self.prob_col_width_fixed = False
        self.id_col_width_fixed = False

        self.mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        # --- Top controls ---
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Enter initial text and press Return…")
        # Size policy: Expanding horizontally so it takes remaining space
        self.input_box.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        # A reasonable minimum so it doesn't collapse
        self.input_box.setMinimumWidth(250)
        self.input_box.returnPressed.connect(self.start_and_preview)

        self.lbl_temp = QLabel("Temperature:")
        # Fixed-size so it does not grab extra space
        self.lbl_temp.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.temp_box = QDoubleSpinBox()
        self.temp_box.setRange(0.0, 2.0)
        self.temp_box.setSingleStep(0.1)  # increment = 0.1
        self.temp_box.setDecimals(1)      # show 1 decimal place
        self.temp_box.setValue(0.0)
        self.temp_box.setMinimumWidth(100)
        # Fixed-size so it does not grab extra space
        self.temp_box.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_reset.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.btn_toggle = QPushButton("Continue")
        self.btn_toggle.clicked.connect(self.toggle_run)
        self.btn_toggle.setEnabled(False)
        self.btn_toggle.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        # --- Token table ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Probability (%)", "Token String", "Token ID"])
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setSelectionMode(self.table.SingleSelection)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(18)
        self.table.verticalHeader().setVisible(False)
        #self.table.horizontalHeader().setStretchLastSection(True)
        # Measure once, then fix widths to reduce reflow cost
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # '#'
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Probability
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Probability        
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Token ID
        self.table.setVerticalScrollMode(self.table.ScrollPerPixel)
        self.table.setHorizontalScrollMode(self.table.ScrollPerPixel)
        # Single-click accepts that token:
        self.table.cellClicked.connect(self.accept_clicked_row)
        # Return key accepts the currently highlighted row:
        self.table.installEventFilter(self)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # --- Layout ---
        top = QHBoxLayout()
        # IMPORTANT: stretch factor on input so it gets remaining space.
        top.addWidget(self.input_box, 1)
        top.addWidget(self.lbl_temp, 0)
        top.addWidget(self.temp_box, 0)
        top.addWidget(self.btn_reset, 0)
        top.addSpacing(12)  # gap between Reset and Continue
        top.addWidget(self.btn_toggle, 0)

        layout = QVBoxLayout()
        layout.addLayout(top, 0)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.output_box, 0)
        layout.setStretch(0, 1)  # controls row
        layout.setStretch(1, 8)  # table (~80%)
        layout.setStretch(2, 2)  # output (~20%)
        self.setLayout(layout)

    @staticmethod
    def apply_uniform_font(app, family="Segoe UI", size=12):
        app.setFont(QFont(family, size))

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.accept_selected_row()
                return True
        return super().eventFilter(obj, event)

    # --- Reset ---
    def reset_all(self):
        if self.worker and self.worker.running:
            self.worker.stop()
            self.worker = None
        self.btn_toggle.setText("Continue")
        self.btn_toggle.setEnabled(False)
        self.input_box.setEnabled(True)
        self.current_text = ""
        self.output_box.clear()
        self.table.setRowCount(0)
        self.prob_col_width_fixed = False
        self.id_col_width_fixed = False

    # --- Busy UX helper ---
    def set_busy(self, yes: bool):
        if yes:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.input_box.setEnabled(False)
            self.btn_toggle.setEnabled(False)
        else:
            QApplication.restoreOverrideCursor()
            self.input_box.setEnabled(True)
            self.btn_toggle.setEnabled(bool(self.current_text))

    # --- Preview step: LLM once, display top-100, highlight/scroll chosen token ---
    def preview_next(self, top_k=TOP_K_DISPLAY):
        T = float(self.temp_box.value())
        out = llm.create_completion(self.current_text, max_tokens=1, temperature=T, logprobs=top_k)

        tok = out["choices"][0]["text"]  # model-chosen next token (given current T)
        logs = out["choices"][0]["logprobs"]["top_logprobs"]
        if logs:
            items = sorted(((tok, math.exp(lp)) for tok, lp in logs[0].items()), key=lambda x: x[1], reverse=True)
        else:
            items = [(tok, 1)]

        return tok, items

    def populate_table(self, items):
        self.table.setUpdatesEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(items))
        for i, (tok, p) in enumerate(items):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))       # '#'
            self.table.setItem(i, 1, QTableWidgetItem(f"{p*100:.4f}"))   # Probability
            tok_item = QTableWidgetItem(display_token_text(tok))         # Token String (quoted + escaped)
            tok_item.setToolTip(token_codepoints_tooltip(tok))
            tok_item.setFont(self.mono_font)                             # monospace for clarity
            tok_item.setData(Qt.UserRole, tok)                           # raw token for logic
            self.table.setItem(i, 2, tok_item)
            self.table.setItem(i, 3, QTableWidgetItem(token_id_repr(tok)))  # Token ID(s)

        # Fix Probability & Token ID column widths once (avoid repeated measurement)
        if not self.prob_col_width_fixed:
            self.table.resizeColumnToContents(1)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
            self.prob_col_width_fixed = True

        if not self.id_col_width_fixed:
            self.table.resizeColumnToContents(3)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
            self.id_col_width_fixed = True

        self.table.setUpdatesEnabled(True)

    def highlight_token(self, tok):
        row = self.find_token_row(tok)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 2), QAbstractItemView.PositionAtCenter)

    def find_token_row(self, tok):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 2)
            if item and item.data(Qt.UserRole) == tok:
                return i
        return None

    def start_and_preview(self):
        self.current_text = self.input_box.text().strip()
        self.output_box.setPlainText(self.current_text)
        # Immediately move focus to the table so the next Return is caught by its handler
        self.table.setFocus(Qt.OtherFocusReason)
        self.set_busy(True)
        self.prev_worker = PreviewWorker(lambda: self.preview_next(TOP_K_DISPLAY))
        self.prev_worker.previewReady.connect(self.on_preview_ready_manual)
        self.prev_worker.finished.connect(lambda: self.set_busy(False))
        self.prev_worker.start()

    def on_preview_ready_manual(self, tok, items):
        self.populate_table(items)
        self.highlight_token(tok)
        self.table.setFocus(Qt.OtherFocusReason)
        self.btn_toggle.setEnabled(True)

    def append_token(self, tok):
        self.current_text += tok
        self.output_box.setPlainText(self.current_text)

    def accept_selected_row(self):      
        row = self.table.currentRow()
        if row < 0:
            return
        tok = self.table.item(row, 2).data(Qt.UserRole)
        self.append_token(tok)
        self.table.setFocus(Qt.OtherFocusReason)
        self.set_busy(True)
        self.prev_worker = PreviewWorker(lambda: self.preview_next(TOP_K_DISPLAY))
        self.prev_worker.previewReady.connect(self.on_preview_ready_manual)
        self.prev_worker.finished.connect(lambda: self.set_busy(False))
        self.prev_worker.start()

    def accept_clicked_row(self, row, _col):
        if row < 0:
            return

        tok = self.table.item(row, 2).data(Qt.UserRole)
        self.append_token(tok)
        self.table.setFocus(Qt.OtherFocusReason)
        self.set_busy(True)
        self.prev_worker = PreviewWorker(lambda: self.preview_next(TOP_K_DISPLAY))
        self.prev_worker.previewReady.connect(self.on_preview_ready_manual)
        self.prev_worker.finished.connect(lambda: self.set_busy(False))
        self.prev_worker.start()

    def toggle_run(self):
        if self.worker and self.worker.running:
            self.worker.stop()
            self.worker = None
            self.btn_toggle.setText("Continue")
            self.input_box.setEnabled(True)
            return
        self.input_box.setEnabled(False)
        self.btn_toggle.setText("Stop")
        self.table.setFocus(Qt.OtherFocusReason)
        self.worker = AutoWorker(lambda: self.preview_next(TOP_K_DISPLAY))
        self.worker.tokenPreview.connect(self.on_auto_preview)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_auto_preview(self, tok, items):
        self.populate_table(items)
        self.highlight_token(tok)
        QApplication.processEvents()  # paint highlight immediately
        self.append_token(tok)
        if self.worker:
            self.worker.ack.set()

    def on_worker_finished(self):
        self.btn_toggle.setText("Continue")
        self.input_box.setEnabled(True)
        self.worker = None


def main():
    app = QApplication(sys.argv)
    TokenPicker.apply_uniform_font(app, "Segoe UI", 12)
    w = TokenPicker()
    w.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()