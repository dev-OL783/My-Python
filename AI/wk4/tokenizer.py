# Import llama before PyQt5
from llama_cpp import Llama

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase

import sys

# ---------------- Model configuration ----------------
# model_name = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
model_name = "/home/yun/dev/uni/220/wk4/llama-3.2-3b-instruct-q4_k_m.gguf"
llm = Llama(model_path=model_name, logits_all=True, verbose=False)

print("Embedding dimension:", llm.n_embd())


# ---------------- Helpers ----------------
def display_token_text(tok: str) -> str:
    """Single-quoted, escaped representation for table display."""
    escaped = tok.encode("unicode_escape").decode("ascii")
    return f"'{escaped}'"

def token_to_piece(token_id: int) -> str:
    """Convert token ID back to its string piece."""
    piece_bytes = llm.detokenize([token_id])
    return piece_bytes.decode("utf-8", errors="replace")


# ---------------- Main Widget ----------------
class TokeniserDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tokenisation Visualiser")

        # Fonts
        self.ui_font = QFont("Segoe UI", 12)
        self.mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        # ---- Top input row ----
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type a phrase and press Return…")
        self.input_box.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.input_box.setMinimumWidth(300)
        self.input_box.returnPressed.connect(self.tokenise)

        self.btn_tokenise = QPushButton("Tokenize")
        self.btn_tokenise.clicked.connect(self.tokenise)
        self.btn_tokenise.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        top = QHBoxLayout()
        top.addWidget(self.input_box, 1)
        top.addWidget(self.btn_tokenise, 0)


        # ---- Token table ----
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Token String", "Token ID"])

        # Make header bold and show underline/border
        header_font = QFont()
        header_font.setBold(True)
        self.table.horizontalHeader().setFont(header_font)
        self.table.setStyleSheet("""
            QHeaderView::section {
                border-bottom: 1px solid #808080;  /* subtle gray line under headers */
                font-weight: bold;
                padding: 4px;
            }
        """)

        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setSelectionMode(self.table.SingleSelection)
        self.table.setVerticalScrollMode(self.table.ScrollPerPixel)
        self.table.setHorizontalScrollMode(self.table.ScrollPerPixel)
        self.table.verticalHeader().setDefaultSectionSize(20)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # '#'
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Token String
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Token ID

        # ---- Output box ----
        self.out = QTextEdit()
        self.out.setReadOnly(True)

        # ---- Main layout ----
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.out, 0)
        layout.setStretch(0, 0)
        layout.setStretch(1, 5)
        layout.setStretch(2, 2)
        self.setLayout(layout)

        self.setFont(self.ui_font)
        self.showMaximized()

    # ---- Core logic ----
    def tokenise(self):
        text = self.input_box.text()
        token_ids = llm.tokenize(text.encode("utf-8"), add_bos=False)

        # Populate table
        self.table.setUpdatesEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(token_ids))

        token_strings_raw = []
        for i, tid in enumerate(token_ids):
            piece = token_to_piece(tid)
            token_strings_raw.append(piece)  # raw for bottom output
            display = display_token_text(piece)

            # Column 0: position
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            # Column 1: token string (monospace)
            item = QTableWidgetItem(display)
            item.setFont(self.mono_font)
            self.table.setItem(i, 1, item)
            # Column 2: token ID
            self.table.setItem(i, 2, QTableWidgetItem(str(tid)))

        self.table.setUpdatesEnabled(True)

        # Output: raw tokens joined by '|', no quotes, no spaces
        self.out.setPlainText("|".join(token_strings_raw))


# ---------------- Entrypoint ----------------
def main():
    app = QApplication(sys.argv)
    w = TokeniserDemo()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()