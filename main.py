import customtkinter as ctk
from app import ANPRApp

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def main():
    root = ctk.CTk()
    root.title("ANPR: Шлагбаум и база номеров")
    root.geometry("1280x720")

    app = ANPRApp(root)

    def _on_close():
        try:
            app.on_close()
        except Exception as e:
            print(f"[ERROR] on_close: {e}")
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    root.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        root.mainloop()
    finally:
        try:
            app.on_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()