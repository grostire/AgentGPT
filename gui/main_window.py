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
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
)
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QBrush, QFont, QKeySequence
from PyQt6.QtCore import Qt, QRectF, QMimeData, QPointF, QLineF
import json # For saving and loading

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

PIN_RADIUS = 5
PIN_DIAMETER = PIN_RADIUS * 2

class BlockPinItem(QGraphicsEllipseItem):
    """
    Represents a connection pin on a DraggableBlockItem.
    """
    def __init__(self, parent_block_item: 'DraggableBlockItem', pin_type: str, pin_index: int, radius: int = PIN_RADIUS):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, parent=parent_block_item)
        self.parent_block_item = parent_block_item
        self.pin_type = pin_type # "input" or "output"
        self.pin_index = pin_index
        self.radius = radius
        self.connections: list[ConnectionItem] = [] # Store connected ConnectionItems

        if self.pin_type == "input":
            self.setBrush(QBrush(Qt.GlobalColor.darkGreen))
        else: # output
            self.setBrush(QBrush(Qt.GlobalColor.darkRed))
        
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

    def get_scene_center_pos(self) -> QPointF:
        """Returns the center of the pin in scene coordinates."""
        # Pin's pos() is its top-left relative to parent. mapToScene converts this.
        # Then add radius to get to the center of the ellipse.
        return self.scenePos() + QPointF(self.radius, self.radius)


    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.pin_type == "output":
            canvas = self.scene().views()[0] # Assuming the first view is our DiagramCanvas
            if isinstance(canvas, DiagramCanvas):
                canvas.start_connection_from_pin(self)
                event.accept()
        # Allow input pins to be selected as end points later, but not start points for now
        # else:
        #     event.ignore() # Input pins don't initiate connections by clicking them directly
        # For now, let parent handle selection or movement if event is not accepted for connection
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent'):
        self.setBrush(QBrush(Qt.GlobalColor.yellow)) # Highlight on hover
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent'):
        if self.pin_type == "input":
            self.setBrush(QBrush(Qt.GlobalColor.darkGreen))
        else: # output
            self.setBrush(QBrush(Qt.GlobalColor.darkRed))
        super().hoverLeaveEvent(event)


class DraggableBlockItem(QGraphicsRectItem):
    """
    Represents a draggable block on the QGraphicsScene.
    """
    def __init__(self, block_type: str, name: str, parent=None):
        super().__init__(parent)
        self.block_type = block_type
        self.block_name = name
        self.backend_block: OpAmpBlock | TFBlock | None = None 
        
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
        # Important for updating connection lines when block moves
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)


        self.text_item = QGraphicsTextItem(f"{self.block_type}\n{self.block_name}", self)
        self.text_item.setDefaultTextColor(Qt.GlobalColor.black)
        self.text_item.setPos(
            self.rect().x() + (self.rect().width() - self.text_item.boundingRect().width()) / 2,
            self.rect().y() + (self.rect().height() - self.text_item.boundingRect().height()) / 2
        )
        
        self.input_pin_items: list[BlockPinItem] = []
        self.output_pin_items: list[BlockPinItem] = []

    def set_backend_block(self, backend_block):
        self.backend_block = backend_block
        self._create_pins() # Call after backend_block is set

    def _create_pins(self):
        if not self.backend_block:
            return

        # Clear existing pins if any (e.g., if block type changes, though not supported yet)
        for pin in self.input_pin_items + self.output_pin_items:
            self.scene().removeItem(pin) # Remove from scene if they were added directly
            # If they are children, they will be removed when parent is removed,
            # but explicit removal is safer if parentage isn't perfectly managed.
        self.input_pin_items.clear()
        self.output_pin_items.clear()

        block_rect = self.rect()
        
        # For TFBlock: 1 input, 1 output
        # For OpAmpBlock: 2 inputs (default), 1 output
        num_inputs = 0
        num_outputs = 0
        if self.backend_block: # Check if backend_block is assigned
            num_inputs = len(self.backend_block.input_pins)
            num_outputs = len(self.backend_block.output_pins)

        # Create Input Pins
        for i in range(num_inputs):
            pin = BlockPinItem(self, "input", i)
            y_pos = (block_rect.height() / (num_inputs + 1)) * (i + 1)
            pin.setPos(block_rect.left() - PIN_RADIUS, y_pos - PIN_RADIUS) # Pin's pos is its top-left
            self.input_pin_items.append(pin)

        # Create Output Pins
        for i in range(num_outputs):
            pin = BlockPinItem(self, "output", i)
            y_pos = (block_rect.height() / (num_outputs + 1)) * (i + 1)
            pin.setPos(block_rect.right() - PIN_RADIUS, y_pos - PIN_RADIUS) # Pin's pos is its top-left
            self.output_pin_items.append(pin)
            
    def itemChange(self, change: QGraphicsRectItem.GraphicsItemChange, value):
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            self.pin_moved()
        return super().itemChange(change, value)

    def pin_moved(self):
        # Update all connections attached to any pin of this block
        for pin_item in self.input_pin_items + self.output_pin_items:
            for connection in pin_item.connections:
                connection.update_position()

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
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 2000, 1500)
        self.setScene(self.scene)
        
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet("background-color: #e0e0e0;")
        
        self.start_connection_pin: BlockPinItem | None = None
        self.temp_connection_line: QGraphicsLineItem | None = None


    def start_connection_from_pin(self, pin_item: BlockPinItem):
        self.start_connection_pin = pin_item
        start_pos = pin_item.get_scene_center_pos()

        self.temp_connection_line = QGraphicsLineItem(QLineF(start_pos, start_pos))
        self.temp_connection_line.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.DashLine))
        self.scene.addItem(self.temp_connection_line)

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
                backend_block_instance = TFBlock(name=item_name, numerator=[1], denominator=[1, 1])
            elif block_type == "OpAmpBlock":
                backend_block_instance = OpAmpBlock(name=item_name, open_loop_gain=1e5, gbw=1e6)
            
            if backend_block_instance:
                gui_block.set_backend_block(backend_block_instance) # This will also create pins
                try:
                    self.main_window.system.add_block(backend_block_instance)
                    self.scene.addItem(gui_block)
                    print(f"Added backend block: {backend_block_instance} with {len(gui_block.input_pin_items)} inputs, {len(gui_block.output_pin_items)} outputs")
                except ValueError as e:
                    QMessageBox.warning(self, "Error", f"Could not add block to system: {e}")
                    BLOCK_COUNTER -=1 
                    return 
            else:
                QMessageBox.warning(self, "Error", f"Unknown block type: {block_type}")
                BLOCK_COUNTER -= 1
                return

            event.acceptProposedAction()
        else:
            event.ignore()

    # def mouseDoubleClickEvent(self, event): # Replaced by context menu
    #     item = self.itemAt(event.pos())
    #     if isinstance(item, DraggableBlockItem):
    #         choices = ["Set as Input", "Set as Output", "Clear Selection"]
    #         # ... (rest of the logic)
    #     super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.temp_connection_line:
            line = self.temp_connection_line.line()
            line.setP2(self.mapToScene(event.pos()))
            self.temp_connection_line.setLine(line)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        if self.temp_connection_line and self.start_connection_pin:
            target_item = self.itemAt(event.scenePos().toPoint()) # Use scenePos for itemAt

            valid_connection = False
            if isinstance(target_item, BlockPinItem) and \
               target_item.pin_type == "input" and \
               target_item.parent_block_item != self.start_connection_pin.parent_block_item:
                
                source_block_backend = self.start_connection_pin.parent_block_item.backend_block
                target_block_backend = target_item.parent_block_item.backend_block
                source_pin_idx = self.start_connection_pin.pin_index
                target_pin_idx = target_item.pin_index

                # Check backend if target pin is already connected
                # Note: self.main_window.system.blocks is dict by name
                target_backend_sys_block = self.main_window.system.blocks.get(target_block_backend.name)
                if target_backend_sys_block and \
                   target_pin_idx < len(target_backend_sys_block.input_pins) and \
                   target_backend_sys_block.input_pins[target_pin_idx] is None:
                    
                    try:
                        self.main_window.system.connect(
                            source_block_backend.name, source_pin_idx,
                            target_block_backend.name, target_pin_idx
                        )
                        # Create visual connection
                        connection_gui = ConnectionItem(self.start_connection_pin, target_item)
                        self.scene.addItem(connection_gui)
                        self.start_connection_pin.connections.append(connection_gui)
                        target_item.connections.append(connection_gui)
                        valid_connection = True
                        print(f"Connected {source_block_backend.name}[{source_pin_idx}] to {target_block_backend.name}[{target_pin_idx}]")
                    except ValueError as e:
                        QMessageBox.warning(self, "Connection Error", f"Could not connect blocks: {e}")
                else:
                    QMessageBox.warning(self, "Connection Error", "Target input pin is already connected or invalid.")
            
            # Cleanup temp line
            self.scene.removeItem(self.temp_connection_line)
            self.temp_connection_line = None
            self.start_connection_pin = None
            if not valid_connection:
                 print("Connection attempt failed or cancelled.")

        super().mouseReleaseEvent(event)


class ConnectionItem(QGraphicsLineItem):
    """
    Represents a visual connection between two BlockPinItems.
    """
    def __init__(self, source_pin: BlockPinItem, dest_pin: BlockPinItem, parent: QGraphicsItem = None):
        super().__init__(parent)
        self.source_pin_item = source_pin
        self.dest_pin_item = dest_pin
        self.setPen(QPen(Qt.GlobalColor.black, 2))
        self.setZValue(-1) # Draw behind blocks and pins
        self.update_position()

    def update_position(self):
        p1 = self.source_pin_item.get_scene_center_pos()
        p2 = self.dest_pin_item.get_scene_center_pos()
        self.setLine(QLineF(p1, p2))


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
        self.current_save_path: str | None = None


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
        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.handle_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.handle_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.handle_save_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def handle_save(self):
        if self.current_save_path:
            self._perform_save(self.current_save_path)
        else:
            self.handle_save_as()

    def handle_save_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save As", self.current_save_path or "", 
                                                   "JSON Files (*.json)")
        if file_path:
            self.current_save_path = file_path
            self._perform_save(file_path)

    def _perform_save(self, file_path: str):
        save_data = {"blocks": [], "connections": []}

        # Gather Block Data
        for backend_block in self.system.blocks.values():
            gui_item = None
            for item in self.diagram_canvas.scene.items():
                if isinstance(item, DraggableBlockItem) and item.block_name == backend_block.name:
                    gui_item = item
                    break
            
            if not gui_item:
                QMessageBox.warning(self, "Save Error", f"Could not find GUI item for backend block {backend_block.name}. Skipping.")
                continue

            block_data = {
                "name": backend_block.name,
                "type": backend_block.__class__.__name__, # "TFBlock" or "OpAmpBlock"
                "gui_position": {"x": gui_item.pos().x(), "y": gui_item.pos().y()}
            }

            if isinstance(backend_block, TFBlock):
                parameters = {
                    "taps": [],
                    "active_tap_index": backend_block.tf.active_tap_index
                }
                for tap in backend_block.tf.taps:
                    parameters["taps"].append({
                        "numerator": tap['num'].tolist(),
                        "denominator": tap['den'].tolist()
                    })
                block_data["parameters"] = parameters
            elif isinstance(backend_block, OpAmpBlock):
                block_data["parameters"] = {
                    "open_loop_gain": backend_block.opamp_model.open_loop_gain,
                    "gbw": backend_block.opamp_model.gbw,
                    "input_impedance": backend_block.opamp_model.input_impedance,
                    "output_impedance": backend_block.opamp_model.output_impedance
                }
            save_data["blocks"].append(block_data)

        # Gather Connection Data (converting tuples to dictionaries)
        for conn_tuple in self.system.connections:
            save_data["connections"].append({
                "source_block_name": conn_tuple[0],
                "source_output_pin_index": conn_tuple[1],
                "dest_block_name": conn_tuple[2],
                "dest_input_pin_index": conn_tuple[3]
            })
        
        try:
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            QMessageBox.information(self, "Save Successful", f"System saved to {file_path}")
        except IOError as e:
            QMessageBox.warning(self, "Save Error", f"Could not save file: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An unexpected error occurred during save: {e}")

    def clear_system(self):
        global BLOCK_COUNTER
        self.diagram_canvas.scene.clear() # Clears all QGraphicsItems
        self.system = System() # Re-initialize the backend system
        self.current_save_path = None
        
        self.selected_input_block_gui_item = None
        self.selected_output_block_gui_item = None
        self.selected_input_block_name = None
        self.selected_output_block_name = None
        
        BLOCK_COUNTER = 0 # Reset global counter for new blocks
        # Any visual selection on blocks is managed by the blocks themselves,
        # so clearing the scene handles their visual state.
        print("System cleared.")


    def handle_open(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", self.current_save_path or "",
                                                   "JSON Files (*.json)")
        if file_path:
            self.clear_system() # Clear current system before loading
            if self._perform_load(file_path):
                self.current_save_path = file_path
                self.setWindowTitle(f"Transfer Function Simulator - {os.path.basename(file_path)}")
            else:
                # If load failed, might want to reset title or leave as is
                self.setWindowTitle("Transfer Function Simulator - Load Failed")


    def _perform_load(self, file_path: str) -> bool:
        global BLOCK_COUNTER
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            QMessageBox.warning(self, "Load Error", f"File not found: {file_path}")
            return False
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Load Error", f"Error decoding JSON file: {e}")
            return False
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred opening file: {e}")
            return False

        max_loaded_id = 0 # To adjust BLOCK_COUNTER after loading

        # Reconstruct Blocks
        try:
            for block_data in loaded_data.get("blocks", []):
                name = block_data["name"]
                block_type = block_data["type"]
                gui_pos = block_data["gui_position"]
                params = block_data.get("parameters", {})

                # Update max_loaded_id from block names like "TFBlock_1", "OpAmpBlock_2"
                if "_" in name:
                    try:
                        num_part = int(name.split("_")[-1])
                        if num_part > max_loaded_id:
                            max_loaded_id = num_part
                    except ValueError:
                        pass # Name doesn't end with _<number>

                backend_block_instance = None
                if block_type == "TFBlock":
                    if not params or "taps" not in params or not params["taps"]:
                        QMessageBox.warning(self, "Load Warning", f"TFBlock '{name}' has no taps data. Using default.")
                        backend_block_instance = TFBlock(name=name, numerator=[1], denominator=[1,1])
                    else:
                        first_tap = params["taps"][0]
                        backend_block_instance = TFBlock(name=name, 
                                                         numerator=first_tap["numerator"], 
                                                         denominator=first_tap["denominator"])
                        for i in range(1, len(params["taps"])):
                            tap_data = params["taps"][i]
                            backend_block_instance.add_tap_to_tf(tap_data["numerator"], tap_data["denominator"])
                        backend_block_instance.set_active_tap_on_tf(params.get("active_tap_index", 0))
                
                elif block_type == "OpAmpBlock":
                    backend_block_instance = OpAmpBlock(
                        name=name,
                        open_loop_gain=params.get("open_loop_gain", 1e5),
                        gbw=params.get("gbw", 1e6),
                        input_impedance=params.get("input_impedance", 1e12),
                        output_impedance=params.get("output_impedance", 50.0)
                    )
                else:
                    QMessageBox.warning(self, "Load Warning", f"Unknown block type '{block_type}' for block '{name}'. Skipping.")
                    continue
                
                self.system.add_block(backend_block_instance)
                
                gui_block = DraggableBlockItem(block_type=block_type, name=name)
                gui_block.backend_block = backend_block_instance
                gui_block.setPos(QPointF(gui_pos["x"], gui_pos["y"])) # QPointF for setPos
                self.diagram_canvas.scene.addItem(gui_block)

            BLOCK_COUNTER = max_loaded_id # Adjust global counter

            # Reconstruct Connections
            for conn_data in loaded_data.get("connections", []):
                self.system.connect(
                    conn_data["source_block_name"],
                    conn_data["source_output_pin_index"],
                    conn_data["dest_block_name"],
                    conn_data["dest_input_pin_index"]
                )
        except KeyError as e:
            QMessageBox.critical(self, "Load Error", f"Missing expected data in save file: {e}")
            self.clear_system() # Clear partially loaded system
            return False
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"An error occurred during loading: {e}")
            self.clear_system() # Clear partially loaded system
            return False

        QMessageBox.information(self, "Load Successful", f"System loaded from {file_path}")
        return True

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
