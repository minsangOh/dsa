"""Entry point for the PyQt-based trading bot application.

The module wires up the Qt UI, configures logging, clears cached files, and
creates the :class:`MainWindow` that controls a background ``TradingBot``.
"""
import sys
import os
import logging
import multiprocessing
import threading
import platform

from pathlib import Path

from ssl_compat import configure_tls

# Align TLS behaviour with certifi so LibreSSL builds work reliably.
configure_tls()

from logging.handlers import RotatingFileHandler
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QTimer

from ui import Ui_MainWindow
from bot import TradingBot
from config import LOG_FILES, CACHE_FILES
from paths import APP_NAME, resolve_data_path

    
def setup_logging():
    """Initialise rotating file handlers for every configured application log."""
    log_level = logging.INFO
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    errors = []

    for log_file_name in LOG_FILES:
        logger_name = log_file_name.split('.')[0]
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        if not logger.handlers:
            try:
                log_file_path = resolve_data_path(log_file_name)
                print(f"Attempting to create log file at: {log_file_path}")  # 경로 출력
                handler = RotatingFileHandler(
                    str(log_file_path),
                    maxBytes=100 * 1024 * 1024,
                    backupCount=5,
                    encoding='utf-8')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                print(f"Handler added for {logger_name}.")
                logger.info(f"--- {log_file_name} log initialized. ---")  # 테스트 로그 기록
            except (OSError, PermissionError) as e:
                error_msg = f"Error setting up logger for {log_file_name}: {e}"
                print(error_msg, file=sys.stderr)
                errors.append(error_msg)
    return errors


class MainWindow(QMainWindow):
    """Main window that presents the UI and routes user intent to the bot."""
    portfolio_request_signal = pyqtSignal()

    def __init__(self):
        """Set up widgets, reload stored credentials, and configure timers."""
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(APP_NAME)

        self.worker = None
        self.portfolio_timer = QTimer(self)
        self.portfolio_timer.timeout.connect(self.request_portfolio_update)
        self.api_key_path = None

        self.connect_signals()
        self.initialize_bot()

    def request_portfolio_update(self):
        """Ask the bot thread to refresh portfolio data when the timer fires."""
        if self.worker and self.worker.isRunning():
            self.portfolio_request_signal.emit()

    def initialize_bot(self):
        """Load cached API keys from disk and populate the input fields."""
        api_key, secret_key = self.load_api_keys()
        if api_key and secret_key:
            self.ui.apiKeyInput.setText(api_key)
            self.ui.secretKeyInput.setText(secret_key)
            path_display = str(self.api_key_path) if self.api_key_path else "api_keys.txt"
            self.ui.log_message("System",
                                f"==============================\nMade by JB KIM\n==============================\n\n\nLoaded API keys.\n\n\n[Method]\n\nPlease enter fixed buy amount.\n\nAfter entering, press the start button.\n\n\n[Glossary of Terms]\n\nPnL = Propit and Loss     ROI = Return on Investment     SL = Stop Loss     Eval. Value = Evaluation amount")
        else:
            self.ui.log_message(
                "System",
                "==============================\nMade by JB KIM\n==============================\n\n\n[Method]\n\nPlease enter API keys and fixed buy amount.\n\nAfter entering, press the start button.\n\n\n[Glossary of Terms]\n\nPnL = Propit and Loss     ROI = Return on Investment     SL = Stop Loss     Eval. Value = Evaluation amount")

    def load_api_keys(self):
        """Read API credentials from known file locations if they exist."""
        self.api_key_path = None

        candidate_paths = [
            resolve_data_path("api_keys.txt"),
            Path(__file__).resolve().parent / "api_keys.txt",
            Path.cwd() / "api_keys.txt",
        ]

        exec_path = Path(sys.executable).resolve() if getattr(sys, "executable", None) else None
        if exec_path:
            candidate_paths.extend([
                exec_path.parent / "api_keys.txt",
                exec_path.parent.parent / "api_keys.txt",
            ])

        if getattr(sys, "_MEIPASS", None):
            candidate_paths.append(Path(sys._MEIPASS) / "api_keys.txt")

        for candidate in candidate_paths:
            try:
                if candidate.exists():
                    with candidate.open('r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) < 2:
                            raise ValueError("File must contain at least two lines.")
                        
                        parts1 = lines[0].strip().split('=', 1)
                        parts2 = lines[1].strip().split('=', 1)

                        if len(parts1) != 2 or len(parts2) != 2:
                            raise ValueError("Each line must be in key=value format.")

                        api_key = parts1[1].strip()
                        secret_key = parts2[1].strip()
                        self.api_key_path = candidate
                        return api_key, secret_key
            except (FileNotFoundError, IndexError, ValueError, PermissionError, OSError) as e:
                self.ui.log_message(
                    "Errors",
                    f"Error reading {candidate}: {e}")
        return None, None

    def connect_signals(self):
        """Wire up the UI controls to their handler methods."""
        self.ui.startButton.clicked.connect(self.start_trading)
        self.ui.stopButton.clicked.connect(self.stop_trading)
        self.ui.allSellButton.clicked.connect(self.all_sell)

    def start_trading(self):
        """Validate credentials and trigger the delayed bot start sequence."""
        api_key = self.ui.apiKeyInput.text()
        secret_key = self.ui.secretKeyInput.text()

        if not api_key or not secret_key:
            QMessageBox.warning(self, "API Key Error",
                                "Both API Key and Secret Key are required.")
            return

        if self.worker is None or not self.worker.isRunning():
            self.ui.statusDisplay.setText("Made by JB Kim")
            self.ui.log_message("System", "Made by JB Kim")
            self.ui.startButton.setEnabled(False)

            QTimer.singleShot(5000, self.proceed_with_start)
        else:
            self.ui.log_message("System", "Bot is already running.")

    def proceed_with_start(self):
        """Instantiate the trading bot and connect signals after the delay."""
        self.ui.startButton.setEnabled(True)

        api_key = self.ui.apiKeyInput.text()
        secret_key = self.ui.secretKeyInput.text()
        fixed_buy_amount_str = self.ui.fixedBuyInput.text()

        self.ui.statusLabel.setText("Running")
        self.worker = TradingBot(api_key, secret_key, fixed_buy_amount_str)

        self.worker.log_signal.connect(self.ui.log_message)
        self.worker.portfolio_signal.connect(self.ui.update_portfolio)
        self.worker.status_signal.connect(self.ui.statusLabel.setText)
        self.worker.status_update_signal.connect(self.ui.statusDisplay.setText)
        self.portfolio_request_signal.connect(
            self.worker.request_portfolio_update)

        self.worker.start()
        self.request_portfolio_update()
        self.portfolio_timer.start(10000)

    def stop_trading(self):
        """Stop the running bot if present and restore the UI to idle state."""
        if self.worker and self.worker.isRunning():
            self.ui.log_message("System", "Requesting to stop auto trading")
            self.portfolio_timer.stop()
            self.worker.stop()
        else:
            self.ui.log_message("System", "No bot is currently running.")
            self.ui.statusLabel.setText("Ready")

    def all_sell(self):
        """Prompt the user and, if confirmed, activate the bot kill switch."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Kill Switch',
                "Are you sure you want to sell all positions and stop the system?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.ui.log_message("System",
                                     "Manually activating Kill Switch.")
                self.portfolio_timer.stop()
                self.worker.activate_kill_switch()
        else:
            self.ui.log_message(
                "System",
                "Cannot activate Kill Switch because no bot is running.")

    def closeEvent(self, event):
        """Ensure the worker thread stops before the window is allowed to close."""
        if self.worker and self.worker.isRunning():
            self.portfolio_timer.stop()
            if self.worker.db:
                self.worker.db.close()
            self.worker.stop()
            self.worker.wait()
        event.accept()


def clear_previous_files():
    """Delete cached files from prior runs to start with a clean state."""
    files_to_clear = [resolve_data_path(file_name) for file_name in CACHE_FILES]
    for file_path in files_to_clear:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Removed old file: {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")


if __name__ == "__main__":
    try:
        if platform.system() != "Windows":
            multiprocessing.set_start_method('fork', force=True)

        print(f"Multiprocessing start method set to: {multiprocessing.get_start_method()}")

    except (RuntimeError, ValueError) as e:
        print(f"Warning: Failed to set multiprocessing start method: {e}", file=sys.stderr)

    clear_previous_files()
    log_errors = setup_logging()
    app = QApplication(sys.argv)

    if log_errors:
        error_message = "Failed to initialize application logging. Please check permissions and paths.\n\n" + "\n".join(
            log_errors)
        QMessageBox.critical(None, "Logging Error", error_message)
        sys.exit(1)

    logging.getLogger("system").info(f"Main GUI thread ID: {threading.get_ident()}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())






