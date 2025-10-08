"""Qt model helpers for presenting portfolio data in a table view."""
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QColor, QBrush


class PortfolioModel(QAbstractTableModel):
    """Table model that exposes portfolio rows to a ``QTableView`` widget."""

    __slots__ = ("_data", "_headers")

    def __init__(self, data=None, headers=None):
        """Store initial rows and column headers for the table model."""
        super().__init__()
        self._data = data or []
        self._headers = headers or []

    def rowCount(self, parent=None):
        """Return the total number of rows available in the model."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Return the number of columns exposed by the model."""
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Provide cell content or alignment information for the given index."""
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row()][index.column()])

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        if role == Qt.ItemDataRole.ForegroundRole:
            if index.column() in [4, 5]:  # PnL and ROI columns
                value = self._data[index.row()][index.column()]
                if isinstance(value, str) and '-' in value:
                    return QBrush(QColor("#E74C3C"))  # Red
                elif isinstance(value, str):
                    return QBrush(QColor("#2ECC71"))  # Green

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        """Return horizontal header labels for display roles."""
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
        return None

    def update_data(self, new_data):
        """Refresh the backing data with a new set of portfolio rows."""
        if len(self._data) != len(new_data):
            self.beginResetModel()
            self._data = new_data
            self.endResetModel()
        else:
            if not new_data:
                return
            self._data = new_data
            top_left = self.index(0, 0)
            bottom_right = self.index(self.rowCount() - 1,
                                      self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right)
