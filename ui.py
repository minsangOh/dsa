"""Utilities that build and update the PyQt6 user interface for the bot."""
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor

from models import PortfolioModel


MAX_LOG_BLOCKS = 1000


class Ui_MainWindow(object):
    """Helper that builds the main window widgets and exposes UI utilities."""

    def setupUi(self, MainWindow):
        """Add all required widgets and layouts to the provided main window."""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1120, 820)

        self.centralWidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralWidget)

        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(24, 24, 24, 24)
        self.mainLayout.setSpacing(20)

        self.topCard = QFrame()
        self.topCard.setObjectName("TopCard")
        self.topCardLayout = QVBoxLayout(self.topCard)
        self.topCardLayout.setContentsMargins(24, 24, 24, 24)
        self.topCardLayout.setSpacing(18)

        self.headerLayout = QHBoxLayout()
        self.headerLayout.setSpacing(12)
        status_font = QFont()
        status_font.setPointSize(20)
        status_font.setWeight(QFont.Weight.Bold)
        self.statusLabel = QLabel("Ready")
        self.statusLabel.setFont(status_font)
        self.headerLayout.addWidget(self.statusLabel)
        self.headerLayout.addStretch()
        self.topCardLayout.addLayout(self.headerLayout)

        self.credentialsLayout = QHBoxLayout()
        self.credentialsLayout.setSpacing(12)

        self.apiKeyInput = QLineEdit()
        self.apiKeyInput.setPlaceholderText("API Key")
        self.apiKeyInput.setEchoMode(QLineEdit.EchoMode.Password)
        self.apiKeyInput.setClearButtonEnabled(True)
        self.apiKeyInput.setMinimumWidth(220)

        self.secretKeyInput = QLineEdit()
        self.secretKeyInput.setPlaceholderText("Secret Key")
        self.secretKeyInput.setEchoMode(QLineEdit.EchoMode.Password)
        self.secretKeyInput.setClearButtonEnabled(True)
        self.secretKeyInput.setMinimumWidth(220)

        self.credentialsLayout.addWidget(self.apiKeyInput)
        self.credentialsLayout.addWidget(self.secretKeyInput)
        self.topCardLayout.addLayout(self.credentialsLayout)

        self.controlLayout = QHBoxLayout()
        self.controlLayout.setSpacing(12)

        self.allSellButton = QPushButton("All Sell")
        self.allSellButton.setObjectName("AllSellButton")

        self.controlLayout.addWidget(self.allSellButton)

        divider = QSpacerItem(12, 12, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.controlLayout.addItem(divider)

        self.fixedBuyLayout = QHBoxLayout()
        self.fixedBuyLayout.setSpacing(8)
        self.fixedBuyLabel = QLabel("Fixed Buy (KRW)")
        self.fixedBuyInput = QLineEdit()
        self.fixedBuyInput.setObjectName("fixedBuyInput")
        self.fixedBuyInput.setPlaceholderText("If left blank, 25% of total assets will be used.")
        self.fixedBuyInput.setClearButtonEnabled(True)
        self.fixedBuyLayout.addWidget(self.fixedBuyLabel)
        self.fixedBuyLayout.addWidget(self.fixedBuyInput)

        self.controlLayout.addLayout(self.fixedBuyLayout)

        self.startButton = QPushButton("Start")
        self.stopButton = QPushButton("Stop")
        self.stopButton.setObjectName("StopButton")

        self.controlLayout.addWidget(self.startButton)
        self.controlLayout.addWidget(self.stopButton)
        self.topCardLayout.addLayout(self.controlLayout)

        self.statusDisplay = QLabel("...")
        self.statusDisplay.setObjectName("StatusDisplay")
        self.statusDisplay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_display_font = QFont()
        status_display_font.setPointSize(12)
        status_display_font.setWeight(QFont.Weight.Medium)
        self.statusDisplay.setFont(status_display_font)
        self.topCardLayout.addWidget(self.statusDisplay)

        self.mainLayout.addWidget(self.topCard)

        self.tabWidget = QTabWidget()
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setMovable(False)
        tab_bar = self.tabWidget.tabBar()
        tab_bar.setObjectName("MainTabBar")
        tab_bar.setDrawBase(False)
        tab_bar.setExpanding(True)
        tab_bar.setUsesScrollButtons(False)
        tab_bar.setElideMode(Qt.TextElideMode.ElideNone)

        self.logDisplays = {}
        self.cumulative_tabs = ["Buy", "Sell", "Trade Fail"]
        self.overwrite_tabs = ["System", "Errors"]

        tabs = ["System", "Portfolio", "Buy", "Sell", "Trade Fail", "Errors"]
        for tab_name in tabs:
            if tab_name == "Portfolio":
                self.portfolioTab = QTableView()
                self.portfolioTab.setAlternatingRowColors(True)
                self.portfolioTab.setObjectName("PortfolioTable")
                self.portfolioTab.verticalHeader().setVisible(False)
                self.portfolioTab.horizontalHeader().setStretchLastSection(True)
                headers = [
                    "Symbol",
                    "Quantity",
                    "Buy Price",
                    "Buy Value",
                    "PnL Amount",
                    "ROI Percent",
                    "SL Price",
                    "Current Price",
                    "Eval. Value",
                ]

                self.portfolio_model = PortfolioModel(data=[], headers=headers)
                self.portfolioTab.setModel(self.portfolio_model)
                self.portfolioTab.horizontalHeader().setSectionResizeMode(
                    QHeaderView.ResizeMode.Stretch
                )

                self.tabWidget.addTab(self.portfolioTab, tab_name)

            elif tab_name in self.overwrite_tabs:
                label = QLabel()
                label.setObjectName("OverwriteLogLabel")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setWordWrap(True)
                self.logDisplays[tab_name] = label
                self.tabWidget.addTab(label, tab_name)

            else:
                plain_text_edit = QPlainTextEdit()
                plain_text_edit.setReadOnly(True)
                log_font = QFont("JetBrains Mono", 10)
                plain_text_edit.setFont(log_font)
                self.logDisplays[tab_name] = plain_text_edit
                self.tabWidget.addTab(plain_text_edit, tab_name)

        self.mainLayout.addWidget(self.tabWidget)

        system_log_label = self.logDisplays["System"]
        font = system_log_label.font()
        font.setPointSize(45)
        system_log_label.setFont(font)

        self._apply_theme(MainWindow)

    def log_message(self, tab, message):
        """Display log output on the requested tab, appending where appropriate."""
        if tab not in self.logDisplays:
            error_message = f"Warning: Attempted to log to an invalid tab: '{tab}'"
            if "Errors" in self.logDisplays:
                self.logDisplays["Errors"].setText(error_message)
            print(error_message)
            return

        log_display = self.logDisplays[tab]

        if tab in self.cumulative_tabs:
            log_display.appendPlainText(message)
            self._trim_log_blocks(log_display)
            log_display.verticalScrollBar().setValue(
                log_display.verticalScrollBar().maximum())
        elif tab in self.overwrite_tabs:
            log_display.setText(message)

    def update_portfolio(self, portfolio_data):
        """Refresh the portfolio table model with the supplied rows."""
        self.portfolio_model.update_data(portfolio_data)

    def _trim_log_blocks(self, plain_text_widget):
        """Cap the number of document blocks so the log widget stays responsive."""
        document = plain_text_widget.document()
        while document.blockCount() > MAX_LOG_BLOCKS:
            cursor = QTextCursor(document.firstBlock())
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _apply_theme(self, MainWindow: QMainWindow) -> None:
        """Apply a polished dark theme across the main window widgets."""
        MainWindow.setStyleSheet(
            """
            #MainWindow {
                background-color: #151925;
                color: #F5F7FF;
            }
            QWidget {
                font-family: 'Pretendard', 'Segoe UI', 'Apple SD Gothic Neo', sans-serif;
                font-size: 15px;
            }
            QFrame#TopCard {
                background-color: #1E2435;
                border-radius: 18px;
            }
            QLabel {
                color: #F5F7FF;
            }
            QLabel#StatusDisplay {
                background-color: rgba(47, 111, 237, 0.12);
                color: #8EB5FF;
                padding: 14px 18px;
                border-radius: 14px;
                letter-spacing: 0.3px;
            }
            QLineEdit {
                background-color: #111827;
                border: 1px solid #27324A;
                border-radius: 12px;
                padding: 10px 14px;
                color: #F5F7FF;
                selection-background-color: #2F6FED;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus {
                border: 1px solid #3B82F6;
                background-color: #0D1524;
            }
            QLineEdit#fixedBuyInput {
                min-width: 330px;
            }
            QPushButton {
                background-color: #3B82F6;
                border-radius: 12px;
                padding: 10px 30px;
                color: #FFFFFF;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4D8EFF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton#StopButton {
                background-color: #E74C3C;
            }
            QPushButton#StopButton:hover {
                background-color: #F0624D;
            }
            QPushButton#StopButton:pressed {
                background-color: #BE3022;
            }
            QPushButton#AllSellButton {
                background-color: #F97316;
            }
            QPushButton#AllSellButton:hover {
                background-color: #FB8A2B;
            }
            QPushButton#AllSellButton:pressed {
                background-color: #EA580C;
            }
            QTabWidget::pane {
                background: #0C042B;
                border: none;
                border-top: 10px solid #0C042B;
            }
            QTabWidget::tab-bar {
                alignment: center;
                padding: 6px 0;
                background-color: #0D1524; 
                border: none;
                margin: 0px;
            }
            QTabWidget {
                background-color: #0D1524;  /* 탭 영역 전체 배경 */
                border: none;
            }
            QTabBar#MainTabBar {
                background-color: #151925;  
                border: none;
                padding: 15px 30px;
                margin: 0 18px;
            }
            QTabBar::tab {
                background-color: #1F2C4A;
                color: #A9B4D4;
                padding: 10px 18px;
                border-radius: 14px;
                border: none;
                margin: 4px 8px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #0C042B;
                color: #FFFFFF;
                border: 1px solid #3B82F6;
                margin: 4px 8px;
            }
            QTabBar::tab:hover {
                color: #D5DCFF;
            }
            QPlainTextEdit {
                background-color: #111827;
                border: 1px solid #27324A;
                border-radius: 14px;
                padding: 14px;
                color: #E2E8F0;
            }
            QLabel#OverwriteLogLabel {
                background-color: #111827;
                border: 1px solid #27324A;
                border-radius: 14px;
                padding: 18px;
                color: #E2E8F0;
            }
            QTableView#PortfolioTable {
                background-color: #111827;
                border: 1px solid #27324A;
                border-radius: 16px;
                gridline-color: #253146;
                selection-background-color: rgba(59, 130, 246, 0.45);
                selection-color: #FFFFFF;
                alternate-background-color: #1A2238;
            }
            QHeaderView { background: transparent; }
            QHeaderView::section {
                background-color: #1F2638;
                color: #E2E8F0;
                padding: 7px;
                border: none;
            }
            QHeaderView::section:horizontal:first { border-top-left-radius: 16px; }
            QHeaderView::section:horizontal:last  { border-top-right-radius: 16px; }
            QTableView::item {
                padding: 6px;
            }
        """
        )
