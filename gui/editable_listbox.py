# ======================
# EditableListbox
# ======================

import tkinter as tk
from tkinter import ttk

class EditableListbox(tk.Listbox):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.edit_item = None
        self.notify_update = None
        self.bind("<Double-1>", self._start_edit)

    def _start_edit(self, event):
        index = self.index(f"@{event.x},{event.y}")
        self.start_edit(index)
        return "break"

    def start_edit(self, index):
        self.edit_item = index
        text = self.get(index)
        y0 = self.bbox(index)[1]
        entry = tk.Entry(self, borderwidth=0, highlightthickness=1)
        entry.bind("<Return>", self.accept_edit)
        entry.bind("<Escape>", self.cancel_edit)

        entry.insert(0, text)
        entry.selection_from(0)
        entry.selection_to("end")
        entry.place(relx=0, y=y0, relwidth=1, width=-1)
        entry.focus_set()
        entry.grab_set()

    def cancel_edit(self, event):
        event.widget.destroy()

    def accept_edit(self, event):
        new_data = event.widget.get()
        new_data = new_data.replace(" ","_")

        self.delete(self.edit_item)
        self.insert(self.edit_item, new_data)
        event.widget.destroy()

        self.notify_update(new_data)
    
    def bind_notify_update(self, callback):
        self.notify_update = callback
