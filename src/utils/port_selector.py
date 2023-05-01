import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk

class PortSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Select COM Port')
        self.ports_list = self.check_which_port()
        self.selected_device = None

        self.create_widgets()

    def check_which_port(self):
        ports = list(serial.tools.list_ports.comports())
        ports_list = []
        for p in ports:
            ports_list.append(p)
        return ports_list

    def create_widgets(self):
        listbox = tk.Listbox(self.root, height=len(self.ports_list), width=50)
        listbox.grid(column=0, row=0)

        for port in self.ports_list:
            listbox.insert(tk.END, f'{port.device}: {port.description}')

        listbox.bind('<<ListboxSelect>>', self.on_select)

        button = ttk.Button(self.root, text='OK', command=self.root.destroy)
        button.grid(column=0, row=1)

    def on_select(self, event):
        selected_index = event.widget.curselection()[0]
        selected_port = self.ports_list[selected_index]
        self.selected_device = selected_port.device

    def run(self):
        self.root.mainloop()
        return self.selected_device


if __name__ == '__main__':
    port_selector = PortSelector()
    selected_device = port_selector.run()
    print(f'Selected port: {selected_device}')