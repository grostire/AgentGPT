import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout, # Will use QVBoxLayout for main content, then QHBoxLayout within
import os # For sys.path modification
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QAbstractItemView,
    QMenuBar,
    QMenu,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QInputDialog,
    QPushButton,
    QMessageBox,
    QDialog,
    QTextEdit,
)
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QRectF, QMimeData, QPointF

# Ensure project root is in sys.path for core.models import
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from core.models import System, TFBlock, OpAmpBlock
except ImportError as e:
    QMessageBox.critical(None, "Import Error", f"Could not import core modules: {e}\nMake sure 'core' directory is in the project root and sys.path is correct.")
    sys.exit(1)


# Global counter for unique block names
BLOCK_COUNTER = 0

class DraggableBlockItem(QGraphicsRectItem):
    """
    Represents a draggable block on the QGraphicsScene.
    """
    def __init__(self, block_type: str, name: str, parent=None):
        super().__init__(parent)
        self.block_type = block_type
        self.block_name = name # Unique name for this instance
        self.backend_block = None # To store the core.models.Block instance
        
        self._is_input = False
        self._is_output = False
        
        self.default_pen = QPen(Qt.GlobalColor.black, 1)
        self.input_pen = QPen(Qt.GlobalColor.green, 3)
        self.output_pen = QPen(Qt.GlobalColor.red, 3)

        self.setRect(0, 0, 120, 70)  # Block size
        self.setBrush(QBrush(QColor("lightblue")))
        self.setPen(self.default_pen)

        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)

        self.text_item = QGraphicsTextItem(f"{self.block_type}\n{self.block_name}", self)
        self.text_item.setDefaultTextColor(Qt.GlobalColor.black)
        rect = self.rect()
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(
            rect.x() + (rect.width() - text_rect.width()) / 2,
            rect.y() + (rect.height() - text_rect.height()) / 2
        )

    def set_selected_as_input(self, selected: bool):
        self._is_input = selected
        self._is_output = False # Cannot be both
        self.update_pen()

    def set_selected_as_output(self, selected: bool):
        self._is_output = selected
        self._is_input = False # Cannot be both
        self.update_pen()
        
    def clear_selection_state(self):
        self._is_input = False
        self._is_output = False
        self.update_pen()

    def update_pen(self):
        if self._is_input:
            self.setPen(self.input_pen)
        elif self._is_output:
            self.setPen(self.output_pen)
        else:
            self.setPen(self.default_pen)
        self.update() # Trigger repaint


class DiagramCanvas(QGraphicsView):
    """
    The canvas area where blocks can be dragged and dropped.
    Inherits from QGraphicsView.
    """
    def __init__(self, main_window, parent=None): # Pass main_window reference
        super().__init__(parent)
        self.main_window = main_window # Store reference to MainWindow
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 2000, 1500)
        self.setScene(self.scene)
        
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet("background-color: #e0e0e0;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            # Check if the text is one of the known block types (optional but good)
            block_type_text = event.mimeData().text()
            if "TFBlock" in block_type_text or "OpAmpBlock" in block_type_text:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        global BLOCK_COUNTER
        if event.mimeData().hasText():
            mime_text = event.mimeData().text()
            # Extract the core block type (e.g., "TFBlock")
            block_type = mime_text.split(" ")[0] 

            drop_position = self.mapToScene(event.pos())
            
            BLOCK_COUNTER += 1
            item_name = f"{block_type}_{BLOCK_COUNTER}"

            gui_block = DraggableBlockItem(block_type=block_type, name=item_name)
            gui_block.setPos(drop_position)
            
            # Link to backend block
            backend_block_instance = None
            if block_type == "TFBlock":
                # Default TF: 1 / (s+1)
                backend_block_instance = TFBlock(name=item_name, numerator=[1], denominator=[1, 1])
            elif block_type == "OpAmpBlock":
                # Default OpAmp: Aol=1e5, GBW=1e6
                backend_block_instance = OpAmpBlock(name=item_name, open_loop_gain=1e5, gbw=1e6)
            
            if backend_block_instance:
                gui_block.backend_block = backend_block_instance
                try:
                    self.main_window.system.add_block(backend_block_instance)
                    self.scene.addItem(gui_block) # Add to scene only if backend add succeeds
                    print(f"Added backend block: {backend_block_instance}") # Debug
                except ValueError as e:
                    QMessageBox.warning(self, "Error", f"Could not add block to system: {e}")
                    BLOCK_COUNTER -=1 # Decrement counter if add failed
                    return # Do not accept drop
            else:
                QMessageBox.warning(self, "Error", f"Unknown block type: {block_type}")
                BLOCK_COUNTER -= 1 # Decrement counter if type unknown
                return

            event.acceptProposedAction()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, DraggableBlockItem):
            choices = ["Set as Input", "Set as Output", "Clear Selection"]
            choice, ok = QInputDialog.getItem(self, "Select Block Role", 
                                              "Choose role for this block:", choices, 0, False)
            if ok and choice:
                if choice == "Set as Input":
                    self.main_window.set_simulation_input_block(item)
                elif choice == "Set as Output":
                    self.main_window.set_simulation_output_block(item)
                elif choice == "Clear Selection":
                    if item == self.main_window.selected_input_block_gui_item:
                        self.main_window.clear_simulation_input_block()
                    if item == self.main_window.selected_output_block_gui_item:
                        self.main_window.clear_simulation_output_block()
                    item.clear_selection_state() # Ensure visual clear if it was neither
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    """
    Main window for the Transfer Function Simulator application.
    """
    def __init__(self):
        super().__init__()

        self.system = System() # Backend system
        self.selected_input_block_gui_item: DraggableBlockItem | None = None
        self.selected_output_block_gui_item: DraggableBlockItem | None = None
        self.selected_input_block_name: str | None = None
        self.selected_output_block_name: str | None = None


        self.setWindowTitle("Transfer Function Simulator")
        self.setGeometry(100, 100, 1200, 800)

        self._create_menu_bar()

        # Main content area: palette on left, canvas on right, button below palette
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Overall layout for the central widget (vertical: top_area, bottom_controls)
        overall_layout = QVBoxLayout(central_widget)

        # Top area for palette and canvas
        top_area_layout = QHBoxLayout()

        # Left panel (palette and button)
        left_panel_layout = QVBoxLayout()

        self.block_palette = QListWidget()
        self.block_palette.addItems([
            "TFBlock (Transfer Function)", 
            "OpAmpBlock (Operational Amplifier)"
        ])
        self.block_palette.setDragEnabled(True)
        # QListWidget default drag mode is SingleSelection, which is fine.
        # Default behavior for QListWidget is to provide text of the item as mimeData.
        self.block_palette.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.block_palette.setMaximumWidth(250)
        left_panel_layout.addWidget(self.block_palette) # Add palette to left panel

        # Run Simulation Button
        self.run_simulation_button = QPushButton("Run Simulation")
        self.run_simulation_button.clicked.connect(self.run_system_simulation)
        self.run_simulation_button.setMaximumWidth(250)
        left_panel_layout.addWidget(self.run_simulation_button) # Add button below palette
        
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)
        left_panel_widget.setMaximumWidth(270) # Give some padding

        top_area_layout.addWidget(left_panel_widget) # Add left panel to top area

        # Block Diagram Canvas (Center/Right Side)
        self.diagram_canvas = DiagramCanvas(self) # Pass self (MainWindow) to canvas
        top_area_layout.addWidget(self.diagram_canvas, 4) # Proportion 4 for canvas

        top_area_widget = QWidget()
        top_area_widget.setLayout(top_area_layout)
        
        overall_layout.addWidget(top_area_widget) # Add top area to overall layout
        # overall_layout could have other things like a status bar later

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def set_simulation_input_block(self, gui_item: DraggableBlockItem):
        if self.selected_input_block_gui_item:
            self.selected_input_block_gui_item.clear_selection_state()
        
        self.selected_input_block_gui_item = gui_item
        self.selected_input_block_name = gui_item.backend_block.name
        gui_item.set_selected_as_input(True)
        print(f"Input block set to: {self.selected_input_block_name}")

    def clear_simulation_input_block(self):
        if self.selected_input_block_gui_item:
            self.selected_input_block_gui_item.clear_selection_state()
        self.selected_input_block_gui_item = None
        self.selected_input_block_name = None
        print("Input block cleared.")

    def set_simulation_output_block(self, gui_item: DraggableBlockItem):
        if self.selected_output_block_gui_item:
            self.selected_output_block_gui_item.clear_selection_state()

        self.selected_output_block_gui_item = gui_item
        self.selected_output_block_name = gui_item.backend_block.name
        gui_item.set_selected_as_output(True)
        print(f"Output block set to: {self.selected_output_block_name}")

    def clear_simulation_output_block(self):
        if self.selected_output_block_gui_item:
            self.selected_output_block_gui_item.clear_selection_state()
        self.selected_output_block_gui_item = None
        self.selected_output_block_name = None
        print("Output block cleared.")

    def run_system_simulation(self):
        if not self.selected_input_block_name or not self.selected_output_block_name:
            QMessageBox.warning(self, "Simulation Error", "Please select both an input and an output block.")
            return

        if self.selected_input_block_name != self.selected_output_block_name:
            QMessageBox.warning(self, "Simulation Error", 
                                "Multi-block simulation with connections is not yet supported. "
                                "Please select the same block as input and output.")
            return

        frequencies = np.logspace(0, 6, 100)  # 1 Hz to 1 MHz

        try:
            magnitudes, phases = self.system.get_system_tf(
                self.selected_input_block_name,
                self.selected_output_block_name,
                frequencies
            )

            # Display results in a dialog
            results_text = "Frequency (Hz) | Magnitude (dB) | Phase (deg)\n"
            results_text += "-" * 60 + "\n"
            for f, m, p in zip(frequencies, magnitudes, phases):
                results_text += f"{f:<15.2e} | {m:<15.2f} | {p:<15.2f}\n"
            
            results_dialog = QDialog(self)
            results_dialog.setWindowTitle("Simulation Results")
            results_dialog.setGeometry(200, 200, 600, 400)
            layout = QVBoxLayout(results_dialog)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFontFamily("Courier") # Monospaced font
            text_edit.setText(results_text)
            layout.addWidget(text_edit)
            results_dialog.exec()

        except ValueError as ve:
            QMessageBox.critical(self, "Simulation Error", f"Error in system configuration or path: {ve}")
        except NotImplementedError as nie:
            QMessageBox.critical(self, "Simulation Error", f"Functionality not implemented: {nie}")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"An unexpected error occurred: {e}")


def main():
    """
    Application entry point.
    """
    app = QApplication(sys.argv)
    
    # It's good practice to set application name and version if distributing
    QApplication.setApplicationName("TransferFunctionSimulator")
    QApplication.setOrganizationName("SciPyTools") # Example
    # QApplication.setApplicationVersion("0.1.0")


    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    # sys.path modification is now at the top of the file.
    main()
