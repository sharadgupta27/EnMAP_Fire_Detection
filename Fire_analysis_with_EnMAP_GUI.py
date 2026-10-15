#!/usr/bin/env python
# coding: utf-8
"""
Desktop GUI for Fire_analysis_with_EnMAP.py.

A sidebar-navigated workspace with four pages that follow the order of the
pipeline itself:

    1  Scene        which EnMAP L1B/L1C radiance product to read, where to write
    2  References   thermal partners and ground measurements to validate against
    3  Retrieval    detection settings, this work's retrieval, the reference
                    method of Amici et al. (2022), the other published methods
    4  Run          the eight pipeline stages ticked off live, the log, summary
                    tiles for both retrievals, and the temperature-map preview

Everything is plain tkinter/ttk with a hand-tuned light and dark theme - no
extra package is needed. The heavy pipeline (GDAL I/O + the per-pixel fits)
runs in a background thread so the window stays responsive; its
stdout/stderr is piped into the on-screen log, with stage, warning, error
and success lines coloured for scannability, and its progress hook drives
the stage list and the progress bar (see run_pipeline()'s progress_hook
contract: hook(i, n, label)).

Run with:  python Fire_analysis_with_EnMAP_GUI.py
"""

import io
import os
import sys
import queue
import contextlib
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import font as tkfont

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import Fire_analysis_with_EnMAP as pipeline
import basemap
import credentials

APP_TITLE = 'EnMAP Wildfire Temperature'
STAGES = list(getattr(pipeline, 'PIPELINE_STAGES', ()))
N_STAGES = len(STAGES) or 8

# ---------------------------------------------------------------------- themes
# Two complete palettes. Every colour the window uses comes from here, so
# switching theme is one re-style pass (see EnMAPFireGUI._apply_theme).
THEMES = {
    'light': {
        'bg': '#f3f5f9',        # page background
        'card': '#ffffff',      # panel surface
        'card_line': '#dfe4ec', # panel border
        'ink': '#18202b',       # primary text
        'muted': '#6b7687',     # secondary text
        'field': '#f8fafc',     # entry background
        'field_line': '#cfd6e0',
        'accent': '#c2410c',    # ember: primary action, active items
        'accent_hi': '#9a330a',
        'accent_soft': '#fdebe3',
        'info': '#2563eb', 'ok': '#15803d', 'warn': '#b45309', 'err': '#b91c1c',
        'ok_soft': '#e6f4ea', 'warn_soft': '#fdf3e2', 'err_soft': '#fdeaea', 'info_soft': '#e8effc',
        'side_bg': '#141a24', 'side_ink': '#e5e9f0', 'side_muted': '#8b95a7',
        'side_hover': '#1f2733', 'side_active': '#2a3341',
        'button': '#e6eaf1', 'button_hi': '#d8deE8'.lower(), 'button_ink': '#18202b',
        'trough': '#e2e7ef',
        'log_bg': '#0f172a', 'log_ink': '#dbe4f0',
        'log_stage': '#7dd3fc', 'log_warn': '#fbbf24', 'log_err': '#f87171', 'log_ok': '#86efac',
        'plot_bg': '#ffffff',
    },
    'dark': {
        'bg': '#0f141b',
        'card': '#161d27',
        'card_line': '#263040',
        'ink': '#e6ebf2',
        'muted': '#93a0b4',
        'field': '#0f151d',
        'field_line': '#334155',
        'accent': '#f0733a',
        'accent_hi': '#ff8a55',
        'accent_soft': '#3a2216',
        'info': '#60a5fa', 'ok': '#4ade80', 'warn': '#fbbf24', 'err': '#f87171',
        'ok_soft': '#12301d', 'warn_soft': '#3a2a0e', 'err_soft': '#3b1414', 'info_soft': '#14233d',
        'side_bg': '#0a0e14', 'side_ink': '#e5e9f0', 'side_muted': '#7c8797',
        'side_hover': '#141b25', 'side_active': '#1e2836',
        'button': '#243040', 'button_hi': '#2e3c4f', 'button_ink': '#e6ebf2',
        'trough': '#1f2836',
        'log_bg': '#0a0e14', 'log_ink': '#d5dde8',
        'log_stage': '#7dd3fc', 'log_warn': '#fbbf24', 'log_err': '#f87171', 'log_ok': '#86efac',
        'plot_bg': '#161d27',
    },
}
FONT = 'Segoe UI'
MONO = 'Consolas'


# ------------------------------------------------------------------ helpers
class _Tooltip:
    """Hover tooltip: a small borderless window near the cursor on <Enter>,
    destroyed on <Leave>. Detail that need not be visible at a glance (full
    explanations, edge cases, format notes) lives here, so the on-screen
    labels stay short."""

    def __init__(self, widget, text, wraplength=440):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip = None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _show(self, _event=None):
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, justify='left', wraplength=self.wraplength,
                 background='#1f2937', foreground='#f3f4f6', relief='flat',
                 font=(FONT, 9), padx=10, pady=7).pack()

    def _hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def add_tooltip(widget, text):
    if text:
        _Tooltip(widget, text)


class _Switch(tk.Canvas):
    """An on/off toggle switch bound to a BooleanVar: a pill track with a
    round knob, accent-coloured when on. Replaces the ttk checkbutton, which
    under the 'clam' theme draws an 'x' in a square."""

    W, H = 40, 22

    def __init__(self, parent, variable, theme_getter, bg_key='card'):
        super().__init__(parent, width=self.W, height=self.H, highlightthickness=0, bd=0,
                         cursor='hand2')
        self.var = variable
        self._theme = theme_getter
        self._bg_key = bg_key
        self.bind('<Button-1>', self.flip)
        self.var.trace_add('write', lambda *_: self.redraw())
        self.redraw()

    def flip(self, _event=None):
        self.var.set(not bool(self.var.get()))

    def redraw(self):
        t = self._theme()
        on = bool(self.var.get())
        self.configure(background=t[self._bg_key])
        self.delete('all')
        track = t['accent'] if on else t['field_line']
        w, h = self.W, self.H
        r = h / 2 - 1
        self.create_oval(1, 1, 2 * r + 1, 2 * r + 1, fill=track, outline=track)
        self.create_oval(w - 2 * r - 1, 1, w - 1, 2 * r + 1, fill=track, outline=track)
        self.create_rectangle(r + 1, 1, w - r - 1, 2 * r + 1, fill=track, outline=track)
        kx = (w - r - 1) if on else (r + 1)
        kr = r - 3
        self.create_oval(kx - kr, r + 1 - kr, kx + kr, r + 1 + kr, fill='#ffffff', outline='#ffffff')


class _AccountsWindow(tk.Toplevel):
    """One window for every archive key and login (credentials.py). Opened
    when a download toggle that needs an account is switched on and that
    account is not set up yet - and again at Run if something is still
    missing - listing only those accounts; or with 'Accounts...' to see and
    change all of them. Logins are written to the netrc file earthaccess
    reads, the FIRMS key to ~/.env; later runs read them from there.
    result: set of services that are now available (None if closed)."""

    FIELDS = {'netrc': [('username', 'Username', False), ('password', 'Password', True)],
              'env': [('map_key', 'MAP_KEY', True)]}

    def __init__(self, parent, theme, services, intro=None):
        super().__init__(parent)
        self.title('Accounts for data downloads')
        self.resizable(False, False)
        self.transient(parent)
        self.configure(background=theme['card'])
        self.result = None
        self.services = list(services)
        body = ttk.Frame(self, padding=18, style='Card.TFrame')
        body.pack(fill='both', expand=True)
        ttk.Label(body, wraplength=520, justify='left', style='Card.TLabel',
                  text=intro or 'These downloads need an account. Enter each once: logins are saved to your '
                                'netrc file and the FIRMS key to ~/.env, and every later run reads them from there.'
                  ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))
        self.vars = {}
        row, first = 1, None
        for s in self.services:
            spec = credentials.SERVICES[s]
            have = credentials.stored(s)
            ttk.Label(body, text=spec['title'], style='CardStrong.TLabel').grid(row=row, column=0, sticky='w',
                                                                                 pady=(8, 0))
            src = credentials.source(s)
            state = ('set by environment variable' if src == 'environment' else
                     f'saved in {credentials.location(s)}' if src == 'file' else 'not set')
            ttk.Label(body, text=f"{spec['used_for']}  ·  {state}", style='CardMuted.TLabel').grid(
                row=row, column=1, columnspan=2, sticky='w', padx=(10, 0), pady=(8, 0))
            row += 1
            link = ttk.Label(body, text=f"No account? {spec['signup']}", style='CardMuted.TLabel', cursor='hand2')
            link.grid(row=row, column=1, columnspan=2, sticky='w', padx=(10, 0))
            link.bind('<Button-1>', lambda _e, u=spec['signup']: __import__('webbrowser').open(u))
            row += 1
            for key, label, secret in self.FIELDS[spec['kind']]:
                ttk.Label(body, text=label, style='Card.TLabel').grid(row=row, column=0, sticky='e', pady=2)
                v = tk.StringVar(value=have.get(key, ''))
                e = ttk.Entry(body, textvariable=v, width=40, show='*' if secret else '')
                e.grid(row=row, column=1, columnspan=2, sticky='w', padx=(10, 0), pady=2)
                self.vars[(s, key)] = v
                first = first or e
                row += 1
        btns = ttk.Frame(body, style='Card.TFrame')
        btns.grid(row=row, column=0, columnspan=3, sticky='e', pady=(14, 0))
        ttk.Button(btns, text='Skip', command=self._on_skip, style='Ghost.TButton').pack(side='right', padx=(8, 0))
        ttk.Button(btns, text='Save', command=self._on_save, style='Accent.TButton').pack(side='right')
        if first is not None:
            first.focus_set()
        self.bind('<Return>', lambda _e: self._on_save())
        self.bind('<Escape>', lambda _e: self._on_skip())
        self.protocol('WM_DELETE_WINDOW', self._on_skip)
        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f'+{max(px, 0)}+{max(py, 0)}')
        self.grab_set()
        parent.wait_window(self)

    def _on_save(self):
        for s in self.services:
            vals = {k: v.get().strip() for (svc, k), v in self.vars.items() if svc == s}
            if all(vals.values()):
                if vals != credentials.stored(s):
                    credentials.save(s, **vals)
            elif any(vals.values()):
                messagebox.showerror('Incomplete', f'{credentials.SERVICES[s]["title"]}: fill in every field '
                                     'or leave all of them empty.', parent=self)
                return
        self.result = {s for s in self.services if credentials.available(s)}
        self.destroy()

    def _on_skip(self):
        self.result = {s for s in self.services if credentials.available(s)}
        self.destroy()


class _ScrollableFrame(ttk.Frame):
    """A frame whose contents scroll vertically when they outgrow the window.
    Put content in `.body`. The scrollbar appears only when needed, so on a
    large screen nothing changes visually."""

    def __init__(self, parent, background):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, takefocus=0, background=background)
        self.vbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.canvas.pack(side='left', fill='both', expand=True)
        self._bar_shown = False
        self.body.bind('<Configure>', self._sync_scrollregion)
        self.canvas.bind('<Configure>', self._stretch_body)
        self.canvas.bind('<Enter>', self._bind_wheel)
        self.canvas.bind('<Leave>', self._unbind_wheel)

    def set_background(self, colour):
        self.canvas.configure(background=colour)

    def _on_scroll_set(self, first, last):
        needed = not (float(first) <= 0.0 and float(last) >= 1.0)
        if needed and not self._bar_shown:
            self.vbar.pack(side='right', fill='y')
            self._bar_shown = True
        elif not needed and self._bar_shown:
            self.vbar.pack_forget()
            self._bar_shown = False
        self.vbar.set(first, last)

    def _sync_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _stretch_body(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, _event=None):
        self.canvas.bind_all('<MouseWheel>', self._on_wheel)
        self.canvas.bind_all('<Button-4>', self._on_wheel)
        self.canvas.bind_all('<Button-5>', self._on_wheel)

    def _unbind_wheel(self, _event=None):
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.canvas.unbind_all(seq)

    def _on_wheel(self, event):
        if not self._bar_shown:
            return
        if getattr(event, 'num', None) == 4:
            delta = -1
        elif getattr(event, 'num', None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
            if abs(event.delta) >= 120:
                delta = int(-event.delta / 120)
        self.canvas.yview_scroll(delta, 'units')


def _check_gdal():
    try:
        from osgeo import gdal
        return True, f'GDAL {getattr(gdal, "__version__", "?")}'
    except Exception as exc:
        return False, f'GDAL missing ({exc.__class__.__name__})'


class QueueWriter(io.TextIOBase):
    """A file-like object that pushes writes onto a queue, so the pipeline's
    print()/tqdm output (on a worker thread) can be rendered by the Tk main
    loop. Subclasses TextIOBase so library code that asks the stream for
    isatty()/encoding/writelines() gets sane answers."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def isatty(self):
        return False

    def writable(self):
        return True

    @property
    def encoding(self):
        return 'utf-8'

    def write(self, text):
        if text:
            self.q.put(('LOG', text))
        return len(text) if text else 0

    def flush(self):
        pass


@contextlib.contextmanager
def _capture_output(q):
    """Send everything written to stdout/stderr in this process to `q` for
    the duration of the block. Process-wide, which is tolerable only because
    the GUI refuses to start a second worker while one is alive."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = QueueWriter(q)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _capturing(q, fn):
    def run():
        with _capture_output(q):
            fn()
    return run


# ---------------------------------------------------------------- the window
class EnMAPFireGUI(tk.Tk):

    NO_COMPARE = 'None (single map)'
    BG_ENMAP = 'EnMAP scene composite'
    BG_S2_AUTO = 'Sentinel-2 cloudless, pre-fire year'
    BACKGROUNDS = [BG_S2_AUTO] + [f'Sentinel-2 cloudless {y}' for y in basemap.YEARS] + [BG_ENMAP]

    RASTER_FILETYPES = [
        ('EnMAP header / raster', '*.hdr *.tif *.tiff *.dat *.bsq *.bil *.bip'),
        ('All files', '*.*'),
    ]
    ENMAP_INPUT_FILETYPES = [
        ('EnMAP metadata / raster', '*METADATA*.xml *.hdr *.tif *.tiff *.dat *.bsq *.bil *.bip'),
        ('EnMAP METADATA.XML', '*METADATA*.xml'),
        ('All files', '*.*'),
    ]
    PAGES = (
        ('scene', '▣', 'Scene', 'Product and output folder'),
        ('retrieval', '⚙', 'Retrieval', 'Detection, methods, uncertainty'),
        ('refs', '◎', 'References', 'Optional: other sensors, ground data'),
        ('run', '▶', 'Run', 'Stages, log, results and map'),
    )

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('1280x820')
        self.minsize(960, 600)
        self._theme_name = 'light'
        self.t = THEMES[self._theme_name]
        self._gdal_ok, self._gdal_msg = _check_gdal()
        self._themed_tk = []       # (widget, {option: palette key}) re-coloured on theme change
        self._switches = []        # _Switch widgets, redrawn on theme change
        self._nav_buttons = {}
        self._pages = {}
        self._current_page = None
        self._stage_rows = []
        self._stage_state = ['todo'] * N_STAGES

        # ---- state variables -------------------------------------------
        self.enmap_path = tk.StringVar()
        self.other_detector_path = tk.StringVar()
        self.merge_vnir_swir = tk.BooleanVar(value=True)
        self.save_merged_cube = tk.BooleanVar(value=True)
        self.results_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'results_temperature'))
        self.save_results = tk.BooleanVar(value=True)
        self.generate_figures = tk.BooleanVar(value=True)

        self.landsat_path = tk.StringVar()
        self.landsat_auto_fetch = tk.BooleanVar(value=False)
        self.ecostress_path = tk.StringVar()
        self.ecostress_auto_fetch = tk.BooleanVar(value=False)
        self.viirs_path = tk.StringVar()
        self.viirs_auto_fetch = tk.BooleanVar(value=False)
        self.slstr_path = tk.StringVar()
        self.slstr_auto_fetch = tk.BooleanVar(value=False)
        self.coregister = tk.BooleanVar(value=True)
        self.field_csv_path = tk.StringVar()
        self.fire_firms = tk.BooleanVar(value=False)
        self.fire_goes = tk.BooleanVar(value=False)
        self.fire_slstr = tk.BooleanVar(value=False)
        self.fire_lsasaf = tk.BooleanVar(value=False)
        self.fire_files = []                    # files on disk: command line only (--fire-partner-file)
        self._accounts = {}
        self._accounts_open = False
        self.fire_window_hours = tk.DoubleVar(value=3.0)
        self.partner_window_hours = tk.DoubleVar(value=12.0)

        self.detection_method = tk.StringVar(value='glrt')
        self.detection_p_fa = tk.StringVar(value='1e-5')
        self.detection_k = tk.DoubleVar(value=6.0)
        self.require_radiance_anomaly = tk.BooleanVar(value=True)
        self.hotspot_margin_px = tk.IntVar(value=5)
        self.run_profile_ml = tk.BooleanVar(value=True)
        self.run_profile_ml_v1 = tk.BooleanVar(value=False)
        self.generate_report = tk.BooleanVar(value=True)
        self.n_trials = tk.IntVar(value=15)
        self.n_bkg_points = tk.IntVar(value=20)
        self.run_all_methods = tk.BooleanVar(value=True)
        self.method_uncertainty = tk.BooleanVar(value=True)
        self.n_monte_carlo = tk.IntVar(value=100)
        self.dennison_transmittance = tk.StringVar(value='template')
        self.apply_smoke_correction = tk.BooleanVar(value=False)
        self.run_parallel = tk.BooleanVar(value=True)
        self.preview_source = tk.StringVar(value='')
        self.compare_source = tk.StringVar(value=self.NO_COMPARE)
        self.compare_split = tk.DoubleVar(value=0.5)
        self._preview_methods = {}      # display label -> normalised per-pixel frame (methods that ran)
        self._compare_state = None      # cached RGBA images of the two compared methods
        self.background = tk.StringVar(value=self.BG_S2_AUTO)
        self._basemaps = {}             # year -> fetched mosaic for the current result
        self._basemap_failed = {}       # year -> error text for the current result
        self._basemap_pending = set()   # years being fetched
        self._acq_dt = None             # acquisition time of the current result (auto year)
        self._scene_dir = None          # results_dir/<scene> of the current result

        self._worker_thread = None
        self._queue = queue.Queue()
        self._last_result = None
        self._last_rgb_composite = None
        self._poll_after_id = None
        self._colorbar = None
        self._cancel_event = None

        # a report needs the result files; v1 is a comparison for v2 only
        self.save_results.trace_add('write', lambda *_: (None if self.save_results.get()
                                                          else self.generate_report.set(False)))
        self.generate_report.trace_add('write', lambda *_: (self.save_results.set(True)
                                                             if self.generate_report.get() else None))
        self.run_profile_ml.trace_add('write', lambda *_: (None if self.run_profile_ml.get()
                                                            else self.run_profile_ml_v1.set(False)))
        self.run_profile_ml_v1.trace_add('write', lambda *_: (self.run_profile_ml.set(True)
                                                               if self.run_profile_ml_v1.get() else None))

        self._build_styles()
        self._build_layout()
        self._apply_theme()
        self._show_page('scene')
        self._poll_queue()

    def destroy(self):
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        super().destroy()

    # ------------------------------------------------------------ styling
    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

    def _apply_theme(self):
        """(Re)configure every ttk style and every themed tk widget from the
        current palette. Called once at start-up and on each toggle."""
        t = self.t
        st = self.style
        self.configure(background=t['bg'])
        base = (FONT, 10)
        st.configure('.', background=t['bg'], foreground=t['ink'], font=base,
                     bordercolor=t['card_line'], troughcolor=t['trough'])
        st.configure('TFrame', background=t['bg'])
        st.configure('TLabel', background=t['bg'], foreground=t['ink'])
        st.configure('Muted.TLabel', background=t['bg'], foreground=t['muted'], font=(FONT, 9))
        st.configure('PageTitle.TLabel', background=t['bg'], foreground=t['ink'], font=(FONT, 20, 'bold'))
        st.configure('PageSub.TLabel', background=t['bg'], foreground=t['muted'], font=(FONT, 10))

        # cards
        st.configure('Card.TFrame', background=t['card'])
        st.configure('CardOuter.TFrame', background=t['card_line'])
        st.configure('Card.TLabel', background=t['card'], foreground=t['ink'])
        # comparison slider: ember thumb on a flat trough (clam draws the thumb in 'background')
        st.configure('Card.Horizontal.TScale', background=t['accent'], troughcolor=t['trough'],
                     bordercolor=t['card'], lightcolor=t['accent'], darkcolor=t['accent'],
                     troughrelief='flat', sliderlength=16, gripcount=0)
        st.map('Card.Horizontal.TScale', background=[('disabled', t['field_line']), ('active', t['accent_hi'])],
               lightcolor=[('disabled', t['field_line'])], darkcolor=[('disabled', t['field_line'])])
        st.configure('Field.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 8, 'bold'))
        # map tool row: quiet text buttons; the active pan/zoom mode is ember-tinted
        st.configure('Tool.TButton', background=t['card'], foreground=t['muted'], bordercolor=t['card'],
                     lightcolor=t['card'], darkcolor=t['card'], relief='flat', padding=(8, 3),
                     focuscolor=t['card'], font=(FONT, 9), width=0)
        st.map('Tool.TButton', background=[('disabled', t['card']), ('active', t['field'])],
               foreground=[('disabled', t['field_line']), ('active', t['ink'])])
        st.configure('ToolOn.TButton', background=t['accent_soft'], foreground=t['accent'],
                     bordercolor=t['accent_soft'], lightcolor=t['accent_soft'], darkcolor=t['accent_soft'],
                     relief='flat', padding=(8, 3), focuscolor=t['accent_soft'], font=(FONT, 9, 'bold'),
                     width=0)
        st.map('ToolOn.TButton', background=[('active', t['accent_soft'])])
        st.configure('CardTitle.TLabel', background=t['card'], foreground=t['ink'], font=(FONT, 11, 'bold'))
        st.configure('CardHint.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 9))
        st.configure('CardStrong.TLabel', background=t['card'], foreground=t['ink'], font=(FONT, 10, 'bold'))
        st.configure('CardMuted.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 8))
        st.configure('Req.TLabel', background=t['card'], foreground=t['accent'], font=(FONT, 10, 'bold'))
        st.configure('Tag.TLabel', background=t['accent_soft'], foreground=t['accent'],
                     font=(FONT, 8, 'bold'), padding=(6, 1))
        st.configure('TagInfo.TLabel', background=t['info_soft'], foreground=t['info'],
                     font=(FONT, 8, 'bold'), padding=(6, 1))
        st.configure('TagWarn.TLabel', background=t['warn_soft'], foreground=t['warn'],
                     font=(FONT, 8, 'bold'), padding=(6, 1))
        st.configure('Card.TCheckbutton', background=t['card'], foreground=t['ink'],
                     indicatorcolor=t['field'], indicatormargin=(0, 0, 6, 0))
        st.map('Card.TCheckbutton', background=[('active', t['card'])],
               indicatorcolor=[('selected', t['accent']), ('!selected', t['field'])])
        st.configure('TCheckbutton', background=t['bg'], foreground=t['ink'])
        st.map('TCheckbutton', background=[('active', t['bg'])])

        # metric tiles
        st.configure('Tile.TFrame', background=t['card'])
        st.configure('TileLabel.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 9))
        st.configure('TileValue.TLabel', background=t['card'], foreground=t['ink'], font=(FONT, 18, 'bold'))
        st.configure('TileValueAccent.TLabel', background=t['card'], foreground=t['accent'], font=(FONT, 18, 'bold'))
        st.configure('TileNote.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 8))

        # stage list
        st.configure('StageTodo.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 9))
        st.configure('StageActive.TLabel', background=t['card'], foreground=t['accent'], font=(FONT, 9, 'bold'))
        st.configure('StageDone.TLabel', background=t['card'], foreground=t['ok'], font=(FONT, 9))
        st.configure('StageSkip.TLabel', background=t['card'], foreground=t['muted'], font=(FONT, 9, 'italic'))

        # inputs
        for w in ('TEntry', 'TSpinbox', 'TCombobox'):
            st.configure(w, fieldbackground=t['field'], background=t['card'],
                         bordercolor=t['field_line'], lightcolor=t['field_line'],
                         darkcolor=t['field_line'], arrowcolor=t['muted'],
                         foreground=t['ink'], insertcolor=t['ink'], padding=5)
            st.map(w, bordercolor=[('focus', t['accent'])],
                   fieldbackground=[('readonly', t['field']), ('disabled', t['bg'])],
                   foreground=[('readonly', t['ink'])])
        self.option_add('*TCombobox*Listbox.background', t['card'])
        self.option_add('*TCombobox*Listbox.foreground', t['ink'])

        # buttons
        st.configure('TButton', padding=(12, 6), background=t['button'], foreground=t['button_ink'],
                     bordercolor=t['button'], relief='flat', focuscolor=t['button'])
        st.map('TButton', background=[('active', t['button_hi']), ('disabled', t['bg'])],
               foreground=[('disabled', t['muted'])])
        st.configure('Accent.TButton', background=t['accent'], foreground='#ffffff',
                     font=(FONT, 10, 'bold'), padding=(20, 9), relief='flat',
                     bordercolor=t['accent'], focuscolor=t['accent'])
        st.map('Accent.TButton', background=[('active', t['accent_hi']), ('disabled', t['accent_soft'])],
               foreground=[('disabled', t['muted'])])
        st.configure('Ghost.TButton', background=t['card'], foreground=t['ink'], bordercolor=t['field_line'],
                     relief='solid', borderwidth=1, padding=(10, 5), focuscolor=t['card'])
        st.map('Ghost.TButton', background=[('active', t['field']), ('disabled', t['card'])],
               foreground=[('disabled', t['muted'])])
        st.configure('Side.TButton', background=t['side_hover'], foreground=t['side_ink'],
                     bordercolor=t['side_active'], relief='flat', padding=(10, 6), focuscolor=t['side_hover'])
        st.map('Side.TButton', background=[('active', t['side_active'])], foreground=[('active', t['side_ink'])])
        st.configure('Danger.TButton', background=t['err_soft'], foreground=t['err'],
                     bordercolor=t['err_soft'], relief='flat', padding=(12, 8), focuscolor=t['err_soft'])
        st.map('Danger.TButton', background=[('active', t['err_soft']), ('disabled', t['bg'])],
               foreground=[('disabled', t['muted'])])

        # misc
        st.configure('Ember.Horizontal.TProgressbar', troughcolor=t['trough'], background=t['accent'],
                     bordercolor=t['trough'], lightcolor=t['accent'], darkcolor=t['accent'], thickness=8)
        st.configure('Vertical.TScrollbar', background=t['button'], troughcolor=t['bg'],
                     bordercolor=t['bg'], arrowcolor=t['muted'], relief='flat')
        st.configure('Horizontal.TScrollbar', background=t['button'], troughcolor=t['bg'],
                     bordercolor=t['bg'], arrowcolor=t['muted'], relief='flat')
        st.configure('TSeparator', background=t['card_line'])
        st.configure('TPanedwindow', background=t['bg'])
        st.configure('Sash', sashthickness=6, gripcount=0, background=t['bg'])

        # plain tk widgets (sidebar, log, canvases) are re-coloured by hand
        for widget, opts in self._themed_tk:
            try:
                widget.configure(**{opt: t[key] for opt, key in opts.items()})
            except tk.TclError:
                pass
        for scroll in getattr(self, '_scrolls', []):
            scroll.set_background(t['bg'])
        for sw in self._switches:
            sw.redraw()
        if hasattr(self, 'log_text'):
            self._configure_log_tags()
        if hasattr(self, 'figure'):
            self._refresh_preview_background()
        self._paint_nav()
        self._paint_stages()

    def _tkthemed(self, widget, **opts):
        """Register a plain tk widget whose colour options follow the palette."""
        self._themed_tk.append((widget, opts))
        widget.configure(**{opt: self.t[key] for opt, key in opts.items()})
        return widget

    def _toggle_theme(self):
        self._theme_name = 'dark' if self._theme_name == 'light' else 'light'
        self.t = THEMES[self._theme_name]
        self.theme_btn.configure(text='☀  Light mode' if self._theme_name == 'dark' else '☽  Dark mode')
        self._apply_theme()

    # ------------------------------------------------------------ layout
    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ---- sidebar ----------------------------------------------------
        side = self._tkthemed(tk.Frame(self, width=252), background='side_bg')
        side.grid(row=0, column=0, sticky='nsw')
        side.grid_propagate(False)
        side.rowconfigure(2, weight=1)

        brand = self._tkthemed(tk.Frame(side), background='side_bg')
        brand.grid(row=0, column=0, sticky='ew', padx=18, pady=(22, 18))
        self._tkthemed(tk.Label(brand, text='\U0001F525', font=(FONT, 22)),
                       background='side_bg', foreground='side_ink').pack(side='left')
        txt = self._tkthemed(tk.Frame(brand), background='side_bg')
        txt.pack(side='left', padx=(8, 0))
        self._tkthemed(tk.Label(txt, text='EnMAP Wildfire', font=(FONT, 12, 'bold'), anchor='w'),
                       background='side_bg', foreground='side_ink').pack(anchor='w')
        self._tkthemed(tk.Label(txt, text='temperature retrieval', font=(FONT, 9), anchor='w'),
                       background='side_bg', foreground='side_muted').pack(anchor='w')

        nav = self._tkthemed(tk.Frame(side), background='side_bg')
        nav.grid(row=1, column=0, sticky='ew', padx=10)
        for i, (key, icon, label, sub) in enumerate(self.PAGES):
            row = self._tkthemed(tk.Frame(nav, cursor='hand2'), background='side_bg')
            row.pack(fill='x', pady=2)
            bar = self._tkthemed(tk.Frame(row, width=3), background='side_bg')
            bar.pack(side='left', fill='y')
            num = self._tkthemed(tk.Label(row, text=f'{i + 1}', font=(FONT, 9, 'bold'), width=2),
                                 background='side_bg', foreground='side_muted')
            num.pack(side='left', padx=(10, 4), pady=8)
            body = self._tkthemed(tk.Frame(row), background='side_bg')
            body.pack(side='left', fill='x', expand=True, pady=6)
            lbl = self._tkthemed(tk.Label(body, text=label, font=(FONT, 10, 'bold'), anchor='w'),
                                 background='side_bg', foreground='side_ink')
            lbl.pack(anchor='w')
            sublbl = self._tkthemed(tk.Label(body, text=sub, font=(FONT, 8), anchor='w'),
                                    background='side_bg', foreground='side_muted')
            sublbl.pack(anchor='w')
            widgets = (row, bar, num, body, lbl, sublbl)
            for w in widgets:
                w.bind('<Button-1>', lambda _e, k=key: self._show_page(k))
                w.bind('<Enter>', lambda _e, k=key: self._paint_nav(hover=k))
                w.bind('<Leave>', lambda _e: self._paint_nav())
            self._nav_buttons[key] = widgets

        foot = self._tkthemed(tk.Frame(side), background='side_bg')
        foot.grid(row=3, column=0, sticky='ew', padx=18, pady=16)
        self.theme_btn = ttk.Button(foot, text='☽  Dark mode', command=self._toggle_theme,
                                    style='Side.TButton')
        self.theme_btn.pack(fill='x')
        gdal_line = self._tkthemed(tk.Frame(foot), background='side_bg')
        gdal_line.pack(fill='x', pady=(10, 0))
        dot = tk.Label(gdal_line, text='●', font=(FONT, 9),
                       foreground=THEMES['light']['ok'] if self._gdal_ok else THEMES['light']['err'])
        self._tkthemed(dot, background='side_bg')
        dot.pack(side='left')
        self._tkthemed(tk.Label(gdal_line, text=self._gdal_msg, font=(FONT, 8)),
                       background='side_bg', foreground='side_muted').pack(side='left', padx=(4, 0))
        about = self._tkthemed(tk.Label(foot, text='About this tool', font=(FONT, 8, 'underline'),
                                        cursor='hand2'), background='side_bg', foreground='side_muted')
        about.pack(anchor='w', pady=(6, 0))
        about.bind('<Button-1>', lambda _e: self._show_about())

        # ---- main column -------------------------------------------------
        main = ttk.Frame(self)
        main.grid(row=0, column=1, sticky='nsew')
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        head = ttk.Frame(main)
        head.grid(row=0, column=0, sticky='ew', padx=28, pady=(22, 8))
        self.page_title = ttk.Label(head, text='', style='PageTitle.TLabel')
        self.page_title.pack(anchor='w')
        self.page_sub = ttk.Label(head, text='', style='PageSub.TLabel')
        self.page_sub.pack(anchor='w', pady=(2, 0))

        self.page_host = ttk.Frame(main)
        self.page_host.grid(row=1, column=0, sticky='nsew')
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)

        self._scrolls = []
        for key, *_ in self.PAGES:
            if key == 'run':
                frame = ttk.Frame(self.page_host)
            else:
                scroll = _ScrollableFrame(self.page_host, background=self.t['bg'])
                self._scrolls.append(scroll)
                frame = scroll
            frame.grid(row=0, column=0, sticky='nsew')
            self._pages[key] = frame
        self._build_scene_page(self._pages['scene'].body)
        self._build_refs_page(self._pages['refs'].body)
        self._build_retrieval_page(self._pages['retrieval'].body)
        self._build_run_page(self._pages['run'])

        self._build_action_bar(main)

    def _show_page(self, key):
        self._current_page = key
        for k, (_key, icon, label, sub) in zip((p[0] for p in self.PAGES), self.PAGES):
            if k == key:
                self.page_title.configure(text=label)
                self.page_sub.configure(text=sub)
        self._pages[key].tkraise()
        self._paint_nav()

    def _paint_nav(self, hover=None):
        t = self.t
        for key, (row, bar, num, body, lbl, sublbl) in self._nav_buttons.items():
            active = key == self._current_page
            bg = t['side_active'] if active else (t['side_hover'] if key == hover else t['side_bg'])
            for w in (row, num, body, lbl, sublbl):
                w.configure(background=bg)
            bar.configure(background=t['accent'] if active else bg)
            num.configure(foreground=t['accent'] if active else t['side_muted'])
            lbl.configure(foreground=t['side_ink'])

    # ------------------------------------------------------------ building blocks
    def _card(self, parent, title, hint=None, tag=None, tag_style='Tag.TLabel', stretch_col=1):
        """A titled panel (1-px border, white surface). Returns the inner
        frame to grid into; content rows start at 2."""
        outer = ttk.Frame(parent, style='CardOuter.TFrame')
        outer.pack(fill='x', padx=28, pady=(0, 14))
        inner = ttk.Frame(outer, style='Card.TFrame', padding=(14, 12, 14, 12))
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        inner.columnconfigure(stretch_col, weight=1)
        head = ttk.Frame(inner, style='Card.TFrame')
        head.grid(row=0, column=0, columnspan=8, sticky='ew', pady=(0, 2))
        ttk.Label(head, text=title, style='CardTitle.TLabel').pack(side='left')
        if tag:
            ttk.Label(head, text=tag, style=tag_style).pack(side='left', padx=(10, 0))
        if hint:
            ttk.Label(inner, text=hint, wraplength=900, justify='left', style='CardHint.TLabel').grid(
                row=1, column=0, columnspan=8, sticky='w', pady=(0, 8))
        return inner

    def _file_row(self, parent, row, label, var, browse_cmd, clearable=True, required=False,
                  tooltip=None, directory=False):
        lbl = ttk.Label(parent, text=label + ('  *' if required else ''),
                        style='Req.TLabel' if required else 'Card.TLabel')
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=6)
        add_tooltip(lbl, tooltip)
        entry = ttk.Entry(parent, textvariable=var, state='readonly')
        entry.grid(row=row, column=1, sticky='ew', pady=6)
        add_tooltip(entry, tooltip)
        ttk.Button(parent, text='Browse', command=browse_cmd, style='Ghost.TButton').grid(
            row=row, column=2, padx=(8, 0), pady=6)
        if clearable:
            ttk.Button(parent, text='✕', width=3, command=lambda: var.set(''),
                       style='Ghost.TButton').grid(row=row, column=3, padx=(4, 0), pady=6)

    def _switch_row(self, parent, text, var, tooltip=None):
        """A toggle switch with its label to the right; clicking either flips
        the variable. Returns the containing frame (pack/grid it yourself)."""
        box = ttk.Frame(parent, style='Card.TFrame')
        sw = _Switch(box, var, lambda: self.t)
        sw.pack(side='left')
        self._switches.append(sw)
        lbl = ttk.Label(box, text=text, style='Card.TLabel', cursor='hand2')
        lbl.pack(side='left', padx=(10, 0))
        lbl.bind('<Button-1>', sw.flip)
        add_tooltip(lbl, tooltip)
        add_tooltip(sw, tooltip)
        return box

    def _toggle(self, parent, row, text, var, tooltip=None, column=0, columnspan=4, pady=(5, 5)):
        box = self._switch_row(parent, text, var, tooltip)
        box.grid(row=row, column=column, columnspan=columnspan, sticky='w', pady=pady)
        return box

    def _spin(self, parent, row, column, label, var, lo, hi, step, tooltip=None):
        lbl = ttk.Label(parent, text=label, style='Card.TLabel')
        lbl.grid(row=row, column=column, sticky='w', padx=(0, 8), pady=6)
        sp = ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=step, width=8)
        sp.grid(row=row, column=column + 1, sticky='w', padx=(0, 24), pady=6)
        add_tooltip(lbl, tooltip)
        add_tooltip(sp, tooltip)
        return sp

    # ------------------------------------------------------------ page: scene
    def _build_scene_page(self, parent):
        card = self._card(
            parent, 'EnMAP product', tag='L1B / L1C radiance',
            hint='Choose the *-METADATA.XML from a stock DLR download: the VNIR and SWIR rasters '
                 'beside it are found and merged automatically. The temperature fit needs '
                 'at-sensor radiance, so an L2A reflectance product will not work.')
        self._file_row(card, 2, 'Product', self.enmap_path, self._browse_enmap, clearable=False,
                       required=True,
                       tooltip='The *-METADATA.XML sidecar (preferred), or a raster file directly: '
                               '.hdr, the VNIR or SWIR spectral-image file, or your own pre-merged '
                               'cube. Must be an L1B/L1C RADIANCE product, not L2A reflectance.')
        self._file_row(card, 3, 'Second detector', self.other_detector_path,
                       self._browse_other_detector,
                       tooltip='Rarely needed: only if the product above is a raster file and its '
                               'VNIR/SWIR sibling is not in the same folder or does not follow the '
                               'DLR naming. Found automatically otherwise.')
        self._toggle(card, 4, 'Merge VNIR + SWIR into one cube', self.merge_vnir_swir,
                     'A stock EnMAP download ships VNIR (~420-1000 nm) and SWIR (~900-2450 nm) as '
                     'two files; with this on, either file or the METADATA.XML is enough and both '
                     'are combined. Turn off to process a single detector file.')
        self._toggle(card, 5, 'Cache the merged cube next to the source files', self.save_merged_cube,
                     'The first merge is saved as an ENVI .dat/.hdr pair beside the inputs, so later '
                     'runs on the same scene skip the merge. Turn off for a read-only folder.')

        out = self._card(parent, 'Output',
                         hint='Results tables (CSV), temperature-map GeoTIFFs, figures and any '
                              'auto-downloaded reference granules are written to a per-scene '
                              'subfolder of this directory.')
        self._file_row(out, 2, 'Results directory', self.results_dir, self._browse_results_dir,
                       clearable=False, directory=True)
        self._toggle(out, 3, 'Write results tables and GeoTIFFs', self.save_results)
        self._toggle(out, 4, 'Write publication figures (PNG)', self.generate_figures)

    # ------------------------------------------------------------ page: references
    def _build_refs_page(self, parent):
        search = self._card(
            parent, 'Find what exists',
            hint='Start here: queries every archive at once (ECOSTRESS, Landsat 8/9, VIIRS LST, VIIRS '
                 'active fire, Sentinel-3 SLSTR) and lists the granules near this scene ranked by time '
                 'offset. Read-only: downloads nothing and needs no login. The time window also limits '
                 'the auto-downloads below.')
        bar = ttk.Frame(search, style='Card.TFrame')
        bar.grid(row=2, column=0, columnspan=6, sticky='w')
        ttk.Button(bar, text='Find thermal partners', command=self._on_find_partners,
                   style='Ghost.TButton').pack(side='left')
        ttk.Label(bar, text='   time window ±', style='Card.TLabel').pack(side='left')
        ttk.Spinbox(bar, from_=1, to=72, increment=1, width=5,
                    textvariable=self.partner_window_hours).pack(side='left', padx=4)
        ttk.Label(bar, text='hours', style='Card.TLabel').pack(side='left')

        card = self._card(
            parent, 'Thermal reference sensors', tag='optional', tag_style='TagInfo.TLabel',
            hint='Independent land-surface-temperature products to check the retrieval against. '
                 'Pick a file, or tick Auto-download to fetch the closest granule in time. All of '
                 'them saturate far below a flaming front, so they validate the background '
                 'temperature field, never the fire core.')
        card.columnconfigure(1, weight=1)
        self._sensor_row(card, 2, 'Landsat 8 / 9', '100 m  ·  ST_B10  ·  Planetary Computer, no login',
                         self.landsat_path, self._browse_landsat, self.landsat_auto_fetch,
                         'A Collection-2 Level-2 ST_B10 surface-temperature GeoTIFF.',
                         'Search and download the closest cloud-free Landsat 8/9 scene from the free '
                         'public Planetary Computer mirror. Ignored when a file is selected.')
        self._sensor_row(card, 3, 'ECOSTRESS', '70 m  ·  ISS, 52°S-52°N  ·  NASA Earthdata login',
                         self.ecostress_path, self._browse_ecostress, self.ecostress_auto_fetch,
                         'An L2(T)_LSTE surface-temperature GeoTIFF from AppEEARS or LP DAAC.',
                         'Search and download the closest ECOSTRESS granule. Needs a free NASA '
                         'Earthdata login; Run prompts for it when none is cached.')
        self._sensor_row(card, 4, 'VIIRS LST', '1 km  ·  SNPP / NOAA-20 / NOAA-21  ·  NASA Earthdata login',
                         self.viirs_path, self._browse_viirs, self.viirs_auto_fetch,
                         'A daily gridded LST granule (VNP21A1D/N, VJ121A1D/N, VJ221A1D/N .h5). The '
                         'observation time is decoded from the View_Time layer, not the date.',
                         'Search NASA Earthdata for the closest VIIRS LST granule. Same login as '
                         'ECOSTRESS; covers every latitude, day and night.')
        self._sensor_row(card, 5, 'Sentinel-3 SLSTR', '1 km  ·  S3A / S3B  ·  Copernicus account (asked once)',
                         self.slstr_path, self._browse_slstr, self.slstr_auto_fetch,
                         'An SL_2_LST product directory (...SEN3); resampled onto the EnMAP '
                         'footprint on load. Saturates at 355.5 K.',
                         'Search the Copernicus Data Space catalogue and download the closest SLSTR '
                         'LST product. The Copernicus Data Space login is asked once and remembered.')
        self._toggle(card, 6, 'Co-register EnMAP to HLS 30 m with AROSICS before comparing (NASA Earthdata login)',
                     self.coregister,
                     'EnMAP L1B is not orthorectified; on the reference scene its positions are ~1 km off. '
                     'Matches the scene against the nearest cloud-free HLS NIR image (AROSICS local tie points, '
                     'affine shift model) so the LST sampling, the FRP matching and the maps use corrected '
                     'positions. Needs arosics (installed in the emit environment). The report shows the shifts and '
                     'an independent check against the LST products.', columnspan=4)

        fire = self._card(
            parent, 'Active-fire products: fire radiative power', tag='comparison', tag_style='TagInfo.TLabel',
            hint='Products that measure the same quantity this work derives - fire radiative power (FRP), '
                 'and for GOES a sub-pixel fire temperature and area. Stage 7 compares their FRP with this '
                 "work's on the shared footprints, with the time difference to EnMAP stated (fire FRP changes "
                 'within minutes; only geostationary slots are close in time). The report has a section on it.',
            stretch_col=5)
        # Every source is downloaded; the ones that need a key or login ask for it once
        # (at Run, or with Sign in) and remember it (credentials.py).
        r = 2
        self._toggle(fire, r, 'FIRMS: VIIRS 375 m and MODIS 1 km detections (free MAP_KEY)',
                     self.fire_firms, 'NASA FIRMS area API. The MAP_KEY is asked once and remembered.',
                     columnspan=4)
        self._account_widgets(fire, r, 'firms')
        self._toggle(fire, r + 1, 'GOES-East / West ABI fire product: FRP, fire temperature and area every '
                                  '10 min (Americas; public, no login)', self.fire_goes,
                     'NOAA public AWS bucket. Outside the GOES disks (Europe, Africa, Asia) it finds nothing and '
                     'says so - use LSA SAF there.', columnspan=6)
        self._toggle(fire, r + 2, 'Sentinel-3 SLSTR FRP (SL_2_FRP): 1 km MWIR, 500 m SWIR at night '
                                  '(Copernicus account)', self.fire_slstr,
                     'Searches and downloads the SL_2_FRP products within the window below. The Copernicus '
                     'login is asked once and remembered.', columnspan=4)
        self._account_widgets(fire, r + 2, 'cdse')
        self._toggle(fire, r + 3, 'LSA SAF SEVIRI FRP-PIXEL: FRP every 15 min (Europe and Africa; LSA SAF account)',
                     self.fire_lsasaf,
                     'Meteosat SEVIRI FRP-PIXEL list products within ±20 min of EnMAP, downloaded from the LSA SAF '
                     'data server. The login is asked once and remembered.', columnspan=4)
        self._account_widgets(fire, r + 3, 'lsasaf')
        self._spin(fire, r + 4, 0, 'Polar-orbiter window (± h)', self.fire_window_hours, 0.5, 48, 0.5,
                   'Polar-orbiter products within this many hours of EnMAP are used (geostationary: ±20 min).')
        # Switching on a download that needs an account opens the Accounts window
        # right away when that account is not set up yet.
        for var, _p, _s, _w in self._login_toggles():
            var.trace_add('write', lambda *_a, v=var: v.get() and self._on_login_toggle())

        ground = self._card(parent, 'Ground measurements', tag='optional', tag_style='TagInfo.TLabel')
        self._file_row(ground, 2, 'Field CSV', self.field_csv_path, self._browse_field_csv,
                       tooltip='A CSV with lat, lon and temperature_K columns; each fire pixel is '
                               'matched to the nearest measurement.')

    def _sensor_row(self, parent, row, name, meta, var, browse_cmd, auto_var, file_tip, auto_tip):
        box = ttk.Frame(parent, style='Card.TFrame')
        box.grid(row=row, column=0, sticky='w', padx=(0, 12), pady=7)
        title = ttk.Label(box, text=name, style='CardStrong.TLabel')
        title.pack(anchor='w')
        add_tooltip(title, file_tip)
        ttk.Label(box, text=meta, style='CardMuted.TLabel').pack(anchor='w')
        entry = ttk.Entry(parent, textvariable=var, state='readonly')
        entry.grid(row=row, column=1, sticky='ew', pady=7)
        add_tooltip(entry, file_tip)
        ttk.Button(parent, text='Browse', command=browse_cmd, style='Ghost.TButton').grid(
            row=row, column=2, padx=(8, 0), pady=7)
        ttk.Button(parent, text='✕', width=3, command=lambda: var.set(''),
                   style='Ghost.TButton').grid(row=row, column=3, padx=(4, 10), pady=7)
        self._switch_row(parent, 'Auto-download', auto_var, auto_tip).grid(
            row=row, column=4, sticky='w', pady=7)

    # ------------------------------------------------------------ page: retrieval
    def _build_retrieval_page(self, parent):
        detect = self._card(
            parent, 'Fire-pixel detection', tag='this work',
            hint='Default "glrt": a scene-wide likelihood-ratio test on every SWIR band with the '
                 "retrieval's own forward model, a background subspace learned from the scene and "
                 'a false-alarm rate per pixel calibrated on the scene itself; a fit-adequacy gate '
                 'separates fires from saturated or specular targets ("candidates", which the '
                 'retrieval then judges). "robust" / "otsu" run the published two-part index test '
                 "(SWIR hotspot index AND radiance anomaly) instead; it also seeds the GLRT's "
                 'fire-free pool.', stretch_col=7)
        ttk.Label(detect, text='Detector', style='Card.TLabel').grid(row=2, column=0, sticky='w', padx=(0, 8), pady=6)
        ttk.Combobox(detect, textvariable=self.detection_method, state='readonly', width=10,
                     values=['glrt', 'robust', 'otsu']).grid(row=2, column=1, sticky='w', padx=(0, 24), pady=6)
        lbl_pfa = ttk.Label(detect, text='False-alarm rate', style='Card.TLabel')
        lbl_pfa.grid(row=2, column=2, sticky='w', padx=(0, 8), pady=6)
        ent_pfa = ttk.Entry(detect, textvariable=self.detection_p_fa, width=8)
        ent_pfa.grid(row=2, column=3, sticky='w', padx=(0, 24), pady=6)
        add_tooltip(lbl_pfa, 'GLRT only: expected false alarms per pixel (1e-5 is about ten per EnMAP tile).')
        add_tooltip(ent_pfa, 'GLRT only: expected false alarms per pixel (1e-5 is about ten per EnMAP tile).')
        self._spin(detect, 2, 4, 'Index sensitivity k', self.detection_k, 1.0, 20.0, 0.5,
                   'Two-part index test (and the GLRT bootstrap): robust standard deviations above '
                   'the scene median. Lower finds more (and fainter) pixels.')
        self._spin(detect, 3, 0, 'Background margin (px)', self.hotspot_margin_px, 0, 50, 1,
                   'Pixels around each detection excluded from the background pool.')
        self._toggle(detect, 4, 'Index test: require the SWIR radiance anomaly as well (recommended)',
                     self.require_radiance_anomaly,
                     'Leave on. The index alone is a ratio whose noise diverges in dark pixels: '
                     'measured on two reference scenes, 70-87% of index-only detections were '
                     'shadow and water. Off reproduces the older behaviour for comparison.',
                     columnspan=8)
        ttk.Label(detect, text='For the index test, "robust" suits scenes where fire is a small fraction '
                               'of the image, the usual case; "otsu" only when the burn area is comparable '
                               'in size to the background.',
                  wraplength=900, justify='left', style='CardHint.TLabel').grid(
            row=5, column=0, columnspan=8, sticky='w', pady=(2, 0))

        prim = self._card(
            parent, 'This work (2026): profile-likelihood retrieval', tag='primary method',
            hint='The retrieval this software publishes, variant 2 (VERIFICATION.md Sections 17, 22-23). '
                 'Per pixel: the L1B quality masks (saturated bands fitted as one-sided constraints), a '
                 'scene fire-term atmospheric transmittance estimated from the adequately fitted fire '
                 'pixels themselves, and a globally optimal profile-likelihood fit with a calibrated error '
                 'model iterated to its fixed point and a 95% confidence interval. Runs first, in stage 4; '
                 'writes its own Results_, Transmittance_ and T_map_ files (..._ProfileMLv2_...).',
            stretch_col=7)
        self._toggle(prim, 2, 'Run this work\'s retrieval', self.run_profile_ml,
                     'About one to two minutes per scene. Turning it off leaves only the reference '
                     'method below.', columnspan=8)
        self._toggle(prim, 3, 'Also run the legacy v1 on the same pixels and compare', self.run_profile_ml_v1,
                     'v1 is the chain as first published: two reweighting passes, and scene-level '
                     'estimates from every converged pixel. It is kept only to reproduce earlier '
                     'results: on synthetic fires the two variants are equally accurate, but on a real '
                     'tile v1\'s scene level was pulled by poorly fitted bright pixels (46 K between '
                     'numerically equivalent variants, against 2 K for v2). Writes its own '
                     '..._ProfileML_... files; adds about one minute.', columnspan=8)
        ttk.Label(prim, text='v2 is the primary method. If v1 is run as well, the report shows the '
                             'difference between the two as a sensitivity of the scene-level estimates.',
                  wraplength=900, justify='left', style='CardHint.TLabel').grid(
            row=4, column=0, columnspan=8, sticky='w', pady=(2, 0))

        ref = self._card(
            parent, 'Amici et al. (2022): multi-trial Planck-mixture fit', tag='reference method',
            tag_style='TagInfo.TLabel',
            hint='The joint nonlinear fit of a Planck curve and a multi-endmember background, as '
                 'published (1400-1900 nm window, water bands blanked, 20 spectra per trial, 15 '
                 'trials, T in 100-2000 K). Always run (stage 5): its '
                 'trial spread, fire-significance test and two-component escalation feed the '
                 'validation and the publication figures, and it is the method this work is '
                 'compared against.', stretch_col=7)
        self._spin(ref, 2, 0, 'Trials per pixel', self.n_trials, 1, 200, 1,
                   'Each pixel is fitted this many times against different random background '
                   'draws; the spread across trials is the reported uncertainty. Runtime scales '
                   'linearly.')
        self._spin(ref, 2, 2, 'Background draws per trial', self.n_bkg_points, 1, 500, 1,
                   'Endmembers drawn from the background pool for each trial.')

        other = self._card(
            parent, 'Other published methods', tag='comparison', tag_style='TagInfo.TLabel',
            hint='Dozier (1981) bi-spectral and Dennison et al. (2006) temperature-library '
                 'retrievals over the same fire pixels, each on its published channels, plus this '
                 "work's estimator through the common driver (no transmittance, no masks), and the "
                 'method-comparison figures (maps, agreement, spectral fit) against the Amici et al. '
                 '(2022) results above. Stage 6, about 30 s for 750 fire pixels.', stretch_col=7)
        self._toggle(other, 2, 'Run Dozier (1981), Dennison et al. (2006) and the comparison figures',
                     self.run_all_methods, columnspan=8)
        ttk.Label(other, text='Dennison et al. (2006) emitted endmembers', style='Card.TLabel').grid(
            row=3, column=0, columnspan=2, sticky='w', pady=6)
        dcb = ttk.Combobox(other, textvariable=self.dennison_transmittance, state='readonly', width=10,
                           values=['template', 'unit'])
        dcb.grid(row=3, column=2, sticky='w', pady=6)
        add_tooltip(dcb, 'template (default): the one-way gas transmittance of this scene on the emitted '
                         'endmembers, the image-based analogue of their MODTRAN step. unit: no atmosphere on the '
                         'emitted term. On the Portugal tile the choice moves Dennison by up to ~170 K '
                         '(VERIFICATION.md S31) - a sensitivity of the published method, reported as such.')

        unc_card = self._card(
            parent, 'Uncertainty analysis', tag='this work', stretch_col=7,
            hint='This work reports a 95% interval on every temperature and on every fire radiative power, and '
                 'a scene budget (Uncertainty_*.csv, report). Dozier (1981) and Dennison et al. (2006) publish no '
                 'uncertainty: with this on, each pixel is re-fitted on noisy replicates of its spectrum (and, '
                 'for Dozier, of its neighbours\' background) and the 95% Monte Carlo interval is added beside '
                 'their unchanged temperature. Amici et al. (2022) keep their published trial spread.')
        self._toggle(unc_card, 2, 'Monte Carlo uncertainty for Dozier (1981) and Dennison et al. (2006)',
                     self.method_uncertainty, columnspan=4)
        self._spin(unc_card, 2, 4, 'Replicates', self.n_monte_carlo, 20, 1000, 10,
                   'Replicates per pixel; 100 resolves a 95% interval to a few kelvin. Runtime scales linearly '
                   '(about 30 s per 100 replicates on 750 pixels and 16 cores).')

        exp = self._card(
            parent, 'Smoke-transmittance correction', tag='experimental', tag_style='TagWarn.TLabel',
            hint='This repository\'s own unpublished candidate (VERIFICATION.md Section 16): a local '
                 'excess smoke/aerosol transmittance from the contrast between each fire pixel\'s '
                 'neighbourhood and the clear-sky background, applied to the fire term only and '
                 'gated behind three significance tests. It fired on none of the real pixels '
                 'checked so far, so expect it to change nothing on most scenes. Writes its own '
                 'comparison CSV.', stretch_col=7)
        self._toggle(exp, 2, 'Also run the smoke-transmittance correction', self.apply_smoke_correction,
                     columnspan=8)

        rep = self._card(
            parent, 'Scene report (PDF)', tag='output', tag_style='TagInfo.TLabel',
            hint='A publication-quality PDF for the scene, written to its results folder as '
                 'Report_<scene>.pdf: statistics of every method that ran (median, quartiles, '
                 '5-95% and full range, temperature classes, model-derived fire radiative power), '
                 'temperature maps of every method on one scale, distributions, pairwise agreement '
                 '(scatter, Bland-Altman, table), this work\'s 95% intervals and fit diagnostics, the '
                 'scene transmittance, the v1/v2 sensitivity, the reference data, methods and caveats, '
                 'the uncertainty budget, every pipeline figure, and a reproducibility record with '
                 'checksums. Needs "Write results tables and GeoTIFFs" (Scene page), which this switch '
                 'turns on; "Other published methods" above gives the full comparison.', stretch_col=7)
        self._toggle(rep, 2, 'Write the PDF report at the end of the run', self.generate_report, columnspan=8)
        ttk.Button(rep, text='Build a report from an existing results folder…', style='Ghost.TButton',
                   command=self._on_build_report).grid(row=3, column=0, columnspan=4, sticky='w', pady=(4, 0))

        perf = self._card(parent, 'Performance', stretch_col=7)
        self._toggle(perf, 2, 'Fit pixels in parallel on all CPU cores', self.run_parallel,
                     'Every per-pixel stage (this work, Amici et al. 2022, the comparison methods and the '
                     'Monte Carlo) runs on all cores; results are identical either way. Off for '
                     'single-pixel debugging.', columnspan=8)

    # ------------------------------------------------------------ page: run
    def _build_run_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # ---- tiles
        tiles = ttk.Frame(parent)
        tiles.grid(row=0, column=0, sticky='ew', padx=28, pady=(0, 12))
        for c in range(6):
            tiles.columnconfigure(c, weight=1, uniform='tile')
        self.tiles = {}
        self._tile_frames = {}
        for c, (key, label, note) in enumerate((
                ('pixels', 'Fire pixels detected', '-'),
                ('profile', 'This work (2026), median T', 'profile-likelihood v2, 95% interval'),
                ('legacy_v1', 'Legacy v1, median T', 'comparison only'),
                ('amici', 'Amici et al. (2022), median T', 'multi-trial Planck-mixture fit'),
                ('quality', 'Fit quality', 'adequate fits / transmittance'),
                ('frp', 'Scene FRP, this work', '95% range (independent - correlated)'))):
            outer = ttk.Frame(tiles, style='CardOuter.TFrame')
            outer.grid(row=0, column=c, sticky='nsew', padx=(0 if c == 0 else 12, 0))
            inner = ttk.Frame(outer, style='Tile.TFrame', padding=(14, 10))
            inner.pack(fill='both', expand=True, padx=1, pady=1)
            ttk.Label(inner, text=label, style='TileLabel.TLabel').pack(anchor='w')
            value = ttk.Label(inner, text='-',
                              style='TileValueAccent.TLabel' if key.startswith('profile') else 'TileValue.TLabel')
            value.pack(anchor='w', pady=(2, 0))
            noteL = ttk.Label(inner, text=note, style='TileNote.TLabel')
            noteL.pack(anchor='w')
            self.tiles[key] = (value, noteL)
            self._tile_frames[key] = outer
        # the legacy v1 tile is shown only after a run that computed v1
        self._tiles_parent = tiles
        self._show_v1_tile(False)

        # ---- stages + log + preview
        body = ttk.Panedwindow(parent, orient='horizontal')
        body.grid(row=2, column=0, sticky='nsew', padx=28, pady=(0, 12))

        left = ttk.Frame(body)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        stages_outer = ttk.Frame(left, style='CardOuter.TFrame')
        stages_outer.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        stages = ttk.Frame(stages_outer, style='Card.TFrame', padding=(14, 10))
        stages.pack(fill='both', expand=True, padx=1, pady=1)
        ttk.Label(stages, text='Pipeline stages', style='CardTitle.TLabel').pack(anchor='w', pady=(0, 4))
        self._stage_rows = []
        for i, name in enumerate(STAGES or [f'Stage {k + 1}' for k in range(N_STAGES)]):
            row = ttk.Frame(stages, style='Card.TFrame')
            row.pack(fill='x', pady=1)
            mark = ttk.Label(row, text='○', width=2, style='StageTodo.TLabel')
            mark.pack(side='left')
            short = name.split(':')[0] if len(name) > 64 else name
            lbl = ttk.Label(row, text=f'{i + 1}. {short}', style='StageTodo.TLabel', wraplength=380,
                            justify='left')
            lbl.pack(side='left', fill='x')
            lbl._tooltip = _Tooltip(lbl, name)
            self._stage_rows.append((mark, lbl))
        # Stage 2's title names the chosen detector; keep it in step with the combobox.
        self.detection_method.trace_add('write', lambda *_: self._label_stages())
        self._label_stages()

        log_outer = ttk.Frame(left, style='CardOuter.TFrame')
        log_outer.grid(row=1, column=0, sticky='nsew')
        log_frame = ttk.Frame(log_outer, style='Card.TFrame', padding=(1, 1))
        log_frame.pack(fill='both', expand=True, padx=1, pady=1)
        log_head = ttk.Frame(log_frame, style='Card.TFrame', padding=(12, 8, 12, 4))
        log_head.pack(fill='x')
        ttk.Label(log_head, text='Log', style='CardTitle.TLabel').pack(side='left')
        ttk.Button(log_head, text='Clear', command=self._clear_log, style='Ghost.TButton').pack(side='right')
        self.log_text = tk.Text(log_frame, state='disabled', height=14, wrap='word', relief='flat',
                                bd=0, padx=12, pady=8)
        self._tkthemed(self.log_text, background='log_bg', foreground='log_ink', insertbackground='log_ink')
        log_bar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_bar.set)
        log_bar.pack(side='right', fill='y')
        self.log_text.pack(fill='both', expand=True)
        self._configure_log_tags()
        body.add(left, weight=3)

        right_outer = ttk.Frame(body, style='CardOuter.TFrame')
        right = ttk.Frame(right_outer, style='Card.TFrame', padding=(16, 12))
        right.pack(fill='both', expand=True, padx=1, pady=1)
        prev_head = ttk.Frame(right, style='Card.TFrame')
        prev_head.pack(fill='x')
        ttk.Label(prev_head, text='Temperature map', style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(prev_head, text='Per-pixel fire temperature over the scene',
                  style='CardHint.TLabel').pack(anchor='w', pady=(1, 0))
        # Method pickers: two equal columns, field caption above each box.
        # The choices are the methods that actually ran (filled in _on_done), never a fixed list.
        pickers = ttk.Frame(right, style='Card.TFrame')
        pickers.pack(fill='x', pady=(10, 0))
        pickers.columnconfigure(0, weight=1, uniform='pick')
        pickers.columnconfigure(1, weight=1, uniform='pick')
        ttk.Label(pickers, text='METHOD', style='Field.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(pickers, text='COMPARE WITH', style='Field.TLabel').grid(row=0, column=1, sticky='w',
                                                                          padx=(10, 0))
        self.preview_combo = ttk.Combobox(pickers, textvariable=self.preview_source, state='disabled',
                                          width=18, values=[])
        self.preview_combo.grid(row=1, column=0, sticky='ew', pady=(3, 0))
        self.preview_combo.bind('<<ComboboxSelected>>', lambda _e: self._render_selected_preview())
        self.compare_combo = ttk.Combobox(pickers, textvariable=self.compare_source, state='disabled',
                                          width=18, values=[self.NO_COMPARE])
        self.compare_combo.grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=(3, 0))
        self.compare_combo.bind('<<ComboboxSelected>>', lambda _e: self._render_selected_preview())
        ttk.Label(pickers, text='BACKGROUND', style='Field.TLabel').grid(row=2, column=0, sticky='w',
                                                                        pady=(8, 0))
        self.background_combo = ttk.Combobox(pickers, textvariable=self.background, state='readonly',
                                             width=18, values=self.BACKGROUNDS)
        self.background_combo.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(3, 0))
        self.background_combo.bind('<<ComboboxSelected>>', lambda _e: self._render_selected_preview())
        # loading / year + licence / geolocation of the background, next to the picker
        self._background_note = ttk.Label(pickers, text='', style='CardMuted.TLabel', justify='left')
        self._background_note.grid(row=4, column=0, columnspan=2, sticky='w', pady=(3, 0))
        pickers.bind('<Configure>', lambda e: self._background_note.configure(
            wraplength=max(e.width - 4, 120)))
        self._preview_pickers = pickers
        # Comparison slider: method A left of the divider, method B right of it, same pixel grid
        # and colour scale; drag the slider to sweep the divider across the fire. The row is
        # shown only while two methods are being compared.
        cmp_row = ttk.Frame(right, style='Card.TFrame')
        self._compare_row = cmp_row
        cmp_labels = ttk.Frame(cmp_row, style='Card.TFrame')
        cmp_labels.pack(fill='x')
        self._compare_a_label = ttk.Label(cmp_labels, text='', style='CardMuted.TLabel')
        self._compare_a_label.pack(side='left')
        self._compare_b_label = ttk.Label(cmp_labels, text='', style='CardMuted.TLabel')
        self._compare_b_label.pack(side='right')
        self.compare_scale = ttk.Scale(cmp_row, from_=0.0, to=1.0, orient='horizontal',
                                       variable=self.compare_split, style='Card.Horizontal.TScale',
                                       command=lambda _v: self._update_compare_split())
        self.compare_scale.pack(fill='x', pady=(2, 0))
        # Map and colorbar get fixed axes. figure.colorbar(ax=...) steals room from
        # the map on every redraw and Colorbar.remove() does not give it back, so
        # each method switch used to shrink the map further.
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_axes([0.02, 0.09, 0.74, 0.83])
        self.ax.set_axis_off()
        self.cax = self.figure.add_axes([0.79, 0.15, 0.035, 0.71])
        self.cax.set_visible(False)
        # The actions row and toolbar are packed (side='bottom') before the canvas:
        # the expanding canvas otherwise pushes them out of a short window.
        actions = ttk.Frame(right, style='Card.TFrame')
        actions.pack(side='bottom', fill='x', pady=(10, 0))
        ttk.Separator(right, orient='horizontal').pack(side='bottom', fill='x', pady=(8, 0))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        # matplotlib's toolbar keeps the view history and the pan/zoom modes, but it is
        # never packed: its stock icons ignore the theme. A themed row drives it instead.
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self._toolbar = tb
        tools = ttk.Frame(right, style='Card.TFrame')
        tools.pack(side='bottom', fill='x', pady=(4, 0))
        self._map_tools = {}
        for key, text, cmd in (('home', '⌂  Fit', tb.home), ('back', '←', tb.back),
                               ('forward', '→', tb.forward),
                               ('pan', '✥  Pan', lambda: self._map_mode('pan')),
                               ('zoom', '⬚  Zoom', lambda: self._map_mode('zoom'))):
            b = ttk.Button(tools, text=text, command=cmd, style='Tool.TButton', state='disabled',
                           takefocus=False)
            b.pack(side='left', padx=(0, 2))
            self._map_tools[key] = b
        # cursor read-out (lon/lat and value) from the hidden toolbar, on one line
        self._map_coords = tk.StringVar(value='')
        tb.message.trace_add('write', lambda *_: self._map_coords.set(
            ' '.join(tb.message.get().split())))
        ttk.Label(tools, textvariable=self._map_coords, style='CardMuted.TLabel',
                  anchor='e').pack(side='right', fill='x', expand=True)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=(8, 0))
        self._tkthemed(self.canvas.get_tk_widget(), background='plot_bg', highlightbackground='plot_bg')
        self.canvas.mpl_connect('scroll_event', self._on_map_scroll)
        self.open_folder_btn = ttk.Button(actions, text='Open folder', state='disabled',
                                          command=self._open_results_folder, style='Ghost.TButton')
        self.open_folder_btn.pack(side='left')
        self.save_plot_btn = ttk.Button(actions, text='Save PNG', state='disabled',
                                        command=self._save_preview, style='Ghost.TButton')
        self.save_plot_btn.pack(side='left', padx=(8, 0))
        self.open_report_btn = ttk.Button(actions, text='Open PDF report', state='disabled',
                                          command=self._open_report, style='Accent.TButton')
        self.open_report_btn.pack(side='right')
        self._report_path = None
        body.add(right_outer, weight=2)
        self._refresh_preview_background()
        self._show_preview_placeholder('Run the pipeline to see\nthe temperature map here.')

    # ------------------------------------------------------------ action bar
    def _build_action_bar(self, parent):
        bar_outer = ttk.Frame(parent, style='CardOuter.TFrame')
        bar_outer.grid(row=2, column=0, sticky='ew')
        bar = ttk.Frame(bar_outer, style='Card.TFrame', padding=(28, 12))
        bar.pack(fill='x', padx=0, pady=(1, 0))
        bar.columnconfigure(3, weight=1)

        self.run_button = ttk.Button(bar, text='▶   Run retrieval', command=self._on_run,
                                     style='Accent.TButton')
        self.run_button.grid(row=0, column=0, rowspan=2, sticky='w')
        self.cancel_button = ttk.Button(bar, text='Cancel', command=self._on_cancel, state='disabled',
                                        style='Danger.TButton')
        self.cancel_button.grid(row=0, column=1, rowspan=2, sticky='w', padx=(10, 22))

        self.stage_label = ttk.Label(bar, text='Ready', style='CardStrong.TLabel')
        self.stage_label.grid(row=0, column=3, sticky='w')
        self.progress = ttk.Progressbar(bar, mode='determinate', maximum=100,
                                        style='Ember.Horizontal.TProgressbar')
        self.progress.grid(row=1, column=3, sticky='ew', pady=(4, 0))
        self.percent_label = ttk.Label(bar, text='', style='CardHint.TLabel', width=16, anchor='e')
        self.percent_label.grid(row=1, column=4, sticky='e', padx=(10, 0))
        self.status_label = ttk.Label(bar, text='●  Idle', style='CardHint.TLabel', anchor='e')
        self.status_label.grid(row=0, column=4, sticky='e', padx=(10, 0))

    # ------------------------------------------------------------ browse handlers
    def _browse_enmap(self):
        path = filedialog.askopenfilename(
            title='Select EnMAP METADATA.XML (preferred) or a radiance raster (VNIR/SWIR)',
            filetypes=self.ENMAP_INPUT_FILETYPES)
        if path:
            self.enmap_path.set(path)

    def _browse_other_detector(self):
        path = filedialog.askopenfilename(title='Select the OTHER detector file (VNIR or SWIR)',
                                           filetypes=self.RASTER_FILETYPES)
        if path:
            self.other_detector_path.set(path)

    def _browse_landsat(self):
        path = filedialog.askopenfilename(title='Select Landsat ST GeoTIFF',
                                           filetypes=[('GeoTIFF', '*.tif *.tiff'), ('All files', '*.*')])
        if path:
            self.landsat_path.set(path)

    def _browse_ecostress(self):
        path = filedialog.askopenfilename(title='Select ECOSTRESS LST GeoTIFF',
                                           filetypes=[('GeoTIFF', '*.tif *.tiff'), ('All files', '*.*')])
        if path:
            self.ecostress_path.set(path)

    def _browse_viirs(self):
        path = filedialog.askopenfilename(
            title='Select VIIRS LST granule (VNP21A1D / VJ121A1D / VJ221A1D .h5)',
            filetypes=[('VIIRS LST granule', '*.h5'), ('All files', '*.*')])
        if path:
            self.viirs_path.set(path)

    def _browse_slstr(self):
        path = filedialog.askdirectory(title='Select Sentinel-3 SLSTR product directory (...SEN3)')
        if path:
            self.slstr_path.set(path)

    def _browse_field_csv(self):
        path = filedialog.askopenfilename(title='Select field measurements CSV',
                                           filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            self.field_csv_path.set(path)

    # ------------------------------------------------------------ archive accounts
    def _login_toggles(self):
        """(switch, file-path var or None, service, what) for every download that needs an account.
        A switch whose file is already chosen downloads nothing, so needs no account."""
        return [(self.ecostress_auto_fetch, self.ecostress_path, 'earthdata', 'ECOSTRESS download'),
                (self.viirs_auto_fetch, self.viirs_path, 'earthdata', 'VIIRS LST download'),
                (self.coregister, None, 'earthdata', 'co-registration (HLS)'),
                (self.slstr_auto_fetch, self.slstr_path, 'cdse', 'SLSTR LST download'),
                (self.fire_slstr, None, 'cdse', 'SLSTR FRP'),
                (self.fire_firms, None, 'firms', 'FIRMS'),
                (self.fire_lsasaf, None, 'lsasaf', 'LSA SAF SEVIRI FRP')]

    def _missing_accounts(self):
        need = []
        for var, path, service, _ in self._login_toggles():
            if var.get() and not (path is not None and path.get()) and service not in need \
                    and not credentials.available(service):
                need.append(service)
        return need

    def _open_accounts(self, services=None, intro=None):
        """The Accounts window for `services` (default: all). Switches whose account is
        still missing afterwards are turned off. Returns the descriptions turned off."""
        if self._accounts_open:
            return []
        self._accounts_open = True
        try:
            _AccountsWindow(self, self.t, services or list(credentials.SERVICES), intro=intro)
        finally:
            self._accounts_open = False
        off = []
        for var, path, service, what in self._login_toggles():
            if var.get() and not (path is not None and path.get()) and not credentials.available(service):
                var.set(False)
                off.append(what)
        for s in self._accounts:
            self._refresh_account(s)
        return off

    def _on_login_toggle(self):
        """A download switch was turned on: ask at once for whatever account is missing."""
        if self._accounts_open or not self._accounts:
            return
        need = self._missing_accounts()
        if need:
            self.after_idle(lambda: self._open_accounts(need))

    def _account_widgets(self, parent, row, service):
        """Where the account comes from + 'Accounts…' for one archive."""
        box = ttk.Frame(parent, style='Card.TFrame')
        box.grid(row=row, column=4, columnspan=3, sticky='w', padx=(8, 0))
        status = tk.StringVar()
        ttk.Label(box, textvariable=status, style='CardMuted.TLabel', width=18).pack(side='left')
        ttk.Button(box, text='Accounts…', style='Ghost.TButton',
                   command=lambda: self._open_accounts()).pack(side='left', padx=(4, 0))
        self._accounts[service] = status
        self._refresh_account(service)

    def _refresh_account(self, service):
        src = credentials.source(service)
        self._accounts[service].set({'file': 'saved ✓', 'environment': 'from environment'}.get(src, 'not set'))

    def _browse_results_dir(self):
        path = filedialog.askdirectory(title='Select results directory')
        if path:
            self.results_dir.set(path)

    # ------------------------------------------------------------ partner search
    def _on_find_partners(self):
        if not self.enmap_path.get():
            messagebox.showerror('Missing input', 'Select an EnMAP product on the Scene page first.')
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            messagebox.showinfo('Busy', 'Wait for the current run to finish first.')
            return
        try:
            window = float(self.partner_window_hours.get())
        except (tk.TclError, ValueError):
            messagebox.showerror('Invalid options', 'The search window must be a number of hours.')
            return
        self._clear_log()
        self._show_page('run')
        self._set_status('running', 'Searching thermal archives')
        self._log('Searching ECOSTRESS, Landsat, VIIRS and Sentinel-3 for partners '
                  'within ±%g h ...\n\n' % window)

        def work():
            try:
                import thermal_partners as tpart
                scene = tpart.enmap_scene_reference(self.enmap_path.get())
                df = tpart.match_thermal_partners(scene, window_hours=window)
                if len(df):
                    show = df[['sensor', 'platform', 'product', 'granule', 'datetime_utc',
                               'dt_hours', 'time_quality', 'day_night', 'overlap']]
                    print()
                    print(show.to_string(index=False))
                    print()
                    print('Rows marked "day-composite" are daily VIIRS products whose catalogue '
                          'timestamp is only a DATE, so their offset is shown as NaN and they are '
                          'sorted last. Load one to resolve its real observation time.')
                else:
                    print('No thermal partner found in any archive for this window.')
                self._queue.put(('PARTNERS', len(df)))
            except ImportError as exc:
                self._queue.put(('ERROR', 'thermal_partners.py could not be imported: %s\n\n'
                                          'It needs h5py and netCDF4 in this environment.' % exc))
            except Exception:
                self._queue.put(('ERROR', traceback.format_exc()))

        self._worker_thread = threading.Thread(target=_capturing(self._queue, work), daemon=True)
        self._worker_thread.start()

    # ------------------------------------------------------------ run / threading
    def _on_run(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        if not self.enmap_path.get():
            self._show_page('scene')
            messagebox.showerror('Missing input', 'Select an EnMAP product first (Scene page).')
            return
        if not os.path.exists(self.enmap_path.get()):
            messagebox.showerror('File not found', f'{self.enmap_path.get()} does not exist.')
            return
        try:
            n_trials = int(self.n_trials.get())
            int(self.n_monte_carlo.get())
            n_bkg = int(self.n_bkg_points.get())
            margin = int(self.hotspot_margin_px.get())
            k = float(self.detection_k.get())
            p_fa = float(self.detection_p_fa.get())
            partner_window = float(self.partner_window_hours.get())
        except (tk.TclError, ValueError):
            messagebox.showerror('Invalid options',
                                 'Trials, background draws, margin, k, the false-alarm rate and the partner window must be numbers.')
            return

        # Accounts are normally entered when a download switch is turned on;
        # anything still missing is asked for here, once, in the same window.
        skipped = []
        need = self._missing_accounts()
        if need:
            skipped = self._open_accounts(need, intro='Before the run: these downloads are switched on but '
                                                      'their account is not set up yet. Enter it once (saved '
                                                      'for every later run), or Skip to run without them.')

        self._set_running(True)
        self._clear_log()
        self._reset_stages()
        self._reset_tiles()
        self._show_page('run')
        self._log(f'Starting pipeline on {self.enmap_path.get()}\n')
        for s in skipped:
            self._log(f'Skipped (no account): {s}.\n')
        self._cancel_event = threading.Event()

        kwargs = dict(
            enmap_radiance_path=self.enmap_path.get(),
            other_detector_path=self.other_detector_path.get() or None,
            merge_vnir_swir=bool(self.merge_vnir_swir.get()),
            save_merged_cube=bool(self.save_merged_cube.get()),
            landsat_lst_path=self.landsat_path.get() or None,
            field_data_csv=self.field_csv_path.get() or None,
            ecostress_path=self.ecostress_path.get() or None,
            ecostress_auto_fetch=bool(self.ecostress_auto_fetch.get()),
            landsat_auto_fetch=bool(self.landsat_auto_fetch.get()),
            viirs_path=self.viirs_path.get() or None,
            viirs_auto_fetch=bool(self.viirs_auto_fetch.get()),
            slstr_path=self.slstr_path.get() or None,
            slstr_auto_fetch=bool(self.slstr_auto_fetch.get()),
            coregister=bool(self.coregister.get()),
            partner_max_time_diff_hours=partner_window,
            results_dir=self.results_dir.get() or 'results_temperature',
            save_results=bool(self.save_results.get()),
            plot_overview=False,
            run_parallel=bool(self.run_parallel.get()),
            profile_device=None if bool(self.run_parallel.get()) else 'serial',
            method_uncertainty=bool(self.method_uncertainty.get()) and bool(self.run_all_methods.get()),
            n_monte_carlo=int(self.n_monte_carlo.get()),
            dennison_transmittance=self.dennison_transmittance.get(),
            hotspot_margin_px=margin,
            detection_method=self.detection_method.get(),
            detection_k=k, detection_p_fa=p_fa,
            require_radiance_anomaly=bool(self.require_radiance_anomaly.get()),
            config={'N_trials': n_trials, 'N_bkg_points': n_bkg},
            progress_hook=self._progress_hook,
            cancel_event=self._cancel_event,
            generate_figures=bool(self.generate_figures.get()),
            extra_methods=('all' if bool(self.run_all_methods.get()) else ()),
            apply_profile_ml=bool(self.run_profile_ml.get()),
            profile_ml_variant='v2',
            apply_profile_ml_v1=bool(self.run_profile_ml.get()) and bool(self.run_profile_ml_v1.get()),
            generate_report=bool(self.generate_report.get()),
            fire_partner_sources=tuple(name for name, var in (('FIRMS', self.fire_firms), ('GOES', self.fire_goes),
                                                              ('SLSTR', self.fire_slstr),
                                                              ('LSASAF', self.fire_lsasaf)) if var.get()),
            fire_partner_files=tuple(self.fire_files),
            fire_partner_window_hours=float(self.fire_window_hours.get()),
            apply_smoke_correction=bool(self.apply_smoke_correction.get()),
        )
        if self.generate_report.get() and not self.save_results.get():
            self._log('Note: the PDF report needs "Write results tables and GeoTIFFs" (Scene page); it is skipped.\n')
        self._worker_thread = threading.Thread(
            target=_capturing(self._queue, lambda: self._worker(**kwargs)), daemon=True)
        self._worker_thread.start()

    def _on_cancel(self):
        if self._cancel_event is None or not (self._worker_thread and self._worker_thread.is_alive()):
            return
        self._cancel_event.set()
        self.cancel_button.config(state='disabled')
        self._set_status('running', 'Cancelling')
        self._log('\nCancel requested - finishing the pixels already in flight, then stopping.\n')

    def _progress_hook(self, i, n, stage=None):
        # Called from the worker thread; queue.Queue is thread-safe. The
        # third argument is the pipeline's stage-aware protocol: a
        # 'Stage k/8: ...' string at each boundary, or the label of the
        # per-pixel loop that is running.
        self._queue.put(('PROGRESS', (i, n, stage)))

    def _worker(self, **kwargs):
        try:
            results_df, scene, temp_map, figure_paths = pipeline.run_pipeline(**kwargs)
            self._queue.put(('DONE', (results_df, scene, temp_map, figure_paths)))
        except pipeline.PipelineCancelled as exc:
            self._queue.put(('CANCELLED', str(exc)))
        except ImportError as exc:
            self._queue.put(('ERROR', 'A required geospatial library is missing: '
                                      f'{exc}\n\nInstall GDAL for your Python environment, e.g.:\n'
                                      '  conda install -c conda-forge gdal\nthen try again.'))
        except Exception:
            self._queue.put(('ERROR', traceback.format_exc()))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'LOG':
                    self._log(payload)
                elif kind == 'PROGRESS':
                    self._on_progress(*payload)
                elif kind == 'DONE':
                    self._on_done(payload)
                elif kind == 'ERROR':
                    self._on_error(payload)
                elif kind == 'CANCELLED':
                    self._on_cancelled(payload)
                elif kind == 'PARTNERS':
                    self._on_partners_found(payload)
                elif kind == 'REPORT':
                    self._on_report_built(payload)
                elif kind == 'BASEMAP':
                    self._on_basemap(*payload)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self._poll_after_id = self.after(100, self._poll_queue)

    # ------------------------------------------------------------ progress / stages
    def _reset_stages(self):
        self._stage_state = ['todo'] * N_STAGES
        self._label_stages()
        self._paint_stages()

    def _label_stages(self):
        if not hasattr(pipeline, 'pipeline_stages'):
            return
        for i, ((_, lbl), name) in enumerate(zip(self._stage_rows,
                                                 pipeline.pipeline_stages(self.detection_method.get()))):
            short = name.split(':')[0] if len(name) > 64 else name
            lbl.configure(text=f'{i + 1}. {short}')
            lbl._tooltip.text = name

    def _paint_stages(self):
        marks = {'todo': ('○', 'StageTodo.TLabel'), 'active': ('◉', 'StageActive.TLabel'),
                 'done': ('●', 'StageDone.TLabel'), 'skip': ('○', 'StageSkip.TLabel')}
        for (mark, lbl), state in zip(self._stage_rows, self._stage_state):
            glyph, style = marks[state]
            mark.configure(text=glyph, style=style)
            lbl.configure(style=style)

    def _on_progress(self, i, n, stage=None):
        if isinstance(stage, str) and stage.startswith('Stage '):
            k = int(i)
            self._stage_state = ['done' if j < k - 1 else ('active' if j == k - 1 else 'todo')
                                 for j in range(N_STAGES)]
            self._paint_stages()
            title = stage.split(':', 1)[1].strip() if ':' in stage else stage
            self.stage_label.configure(text=f'Stage {k}/{n}  ·  {title}')
            self.progress.configure(maximum=100, value=0)
            self.percent_label.configure(text='')
            return
        self.progress.configure(maximum=max(n, 1), value=i)
        self.percent_label.configure(text=f'{i}/{n}  ({100 * i / max(n, 1):.0f}%)')
        if stage:
            self.stage_label.configure(text=f'{stage}')

    def _on_partners_found(self, count):
        if count:
            self._set_status('done', f'{count} thermal partner granule(s) found')
        else:
            self._set_status('idle', 'No thermal partners in this window')

    # ------------------------------------------------------------ completion
    def _reset_tiles(self):
        for value, note in self.tiles.values():
            value.configure(text='-')
        self.tiles['pixels'][1].configure(text='-')
        self.tiles['profile'][1].configure(text='profile-likelihood, 95% interval')
        self.tiles['legacy_v1'][1].configure(text='comparison only')
        self._report_path = None
        self.open_report_btn.configure(state='disabled')
        self.tiles['amici'][1].configure(text='multi-trial Planck-mixture fit')
        self.tiles['quality'][1].configure(text='adequate fits / transmittance')
        self.tiles['frp'][1].configure(text='95% range (independent - correlated)')
        self._preview_methods = {}
        self._compare_state = None
        self.preview_combo.configure(state='disabled', values=[])
        self.preview_source.set('')
        self.compare_combo.configure(state='disabled', values=[self.NO_COMPARE])
        self.compare_source.set(self.NO_COMPARE)
        self.compare_scale.state(['disabled'])
        self._show_compare_row(False)
        self._background_note.configure(text='')
        self._scene_dir = None

    def _on_done(self, payload):
        results_df, scene, temp_map, figure_paths = payload
        self._last_result = payload
        self._basemaps, self._basemap_failed, self._basemap_pending = {}, {}, set()
        self._scene_dir = (getattr(results_df, 'attrs', {}) or {}).get('results_dir')
        try:
            self._acq_dt = pipeline._scene_acquisition_datetime(self.enmap_path.get())
        except Exception:
            self._acq_dt = None
        try:
            self._last_rgb_composite = pipeline._build_rgb_composite(scene)
        except Exception:
            self._last_rgb_composite = None
            self._log('\nCould not build a background composite for the preview (showing '
                      'temperature only).\n')
        self._set_running(False)
        self._stage_state = ['done'] * N_STAGES
        self._paint_stages()
        self.stage_label.configure(text='Finished')

        n_fire = len(results_df)
        amici_med = float(np.nanmedian(results_df['median_temp_K'])) if n_fire else float('nan')
        n_at_bound = int(results_df['T_at_bound'].sum()) if 'T_at_bound' in results_df else 0
        n_escalated = (int(results_df['escalated_two_component'].sum())
                       if 'escalated_two_component' in results_df else 0)

        attrs = getattr(results_df, 'attrs', {}) or {}
        pml = attrs.get('profile_ml')
        pdf = pml.get('df') if isinstance(pml, dict) else None
        self.tiles['pixels'][0].configure(text=f'{n_fire}')
        self.tiles['pixels'][1].configure(text=f'{n_at_bound} at the search bound' if n_at_bound
                                          else f'{self.detection_method.get()} detector')
        self.tiles['amici'][0].configure(text=f'{amici_med:.0f} K' if np.isfinite(amici_med) else '-')
        self.tiles['amici'][1].configure(
            text=(f'{n_escalated} escalated to two components' if n_escalated
                  else 'multi-trial Planck-mixture fit'))
        summary = f'{n_fire} fire pixels; Amici et al. (2022) median {amici_med:.0f} K'
        if pdf is not None and len(pdf):
            ok = (pdf['fit_quality'].values == 'ok') & np.isfinite(pdf['T_K'].values)
            fin = ok & ~(pdf['T_lower_unbounded'].values | pdf['T_upper_unbounded'].values)
            pmed = float(np.nanmedian(pdf['T_K'].values[ok])) if ok.any() else float('nan')
            width = float(np.nanmedian((pdf['T_ci_high'] - pdf['T_ci_low']).values[fin])) if fin.any() else float('nan')
            self.tiles['profile'][0].configure(text=f'{pmed:.0f} K' if np.isfinite(pmed) else '-')
            self.tiles['profile'][1].configure(
                text=(f'median 95% interval ±{width / 2:.0f} K' if np.isfinite(width)
                      else 'no finite interval'))
            n_cens = int((pdf['n_bands_censored'] > 0).sum()) if 'n_bands_censored' in pdf else 0
            self.tiles['quality'][0].configure(text=f'{int(ok.sum())}/{len(pdf)}')
            tau_txt = 'transmittance applied' if pml.get('tau_applied') else 'no transmittance (< 3 px)'
            self.tiles['quality'][1].configure(text=f'{tau_txt}; {n_cens} px saturated')
            summary += f'; this work median {pmed:.0f} K'
            if 'FRP_MW' in pdf:
                v = ok & np.isfinite(pdf['FRP_MW'].values)
                tot = float(np.nansum(pdf['FRP_MW'].values[v]))
                lo_, hi_ = pdf['FRP_ci_low_MW'].values[v], pdf['FRP_ci_high_MW'].values[v]
                fin = np.isfinite(lo_) & np.isfinite(hi_)
                ind = 1.96 * np.sqrt(np.sum(((hi_ - lo_)[fin] / 3.92) ** 2))
                self.tiles['frp'][0].configure(text=f'{tot:.0f} MW')
                self.tiles['frp'][1].configure(
                    text=f'±{ind:.0f} (indep.); {np.sum(lo_[fin]):.0f}-{np.sum(hi_[fin]):.0f} MW (corr.)')
                sat_col = ('FRP_min_joint_MW_saturated' if 'FRP_min_joint_MW_saturated' in pdf
                           else 'FRP_min_MW_saturated')
                if sat_col in pdf:
                    lb = np.nansum(pdf[sat_col].values[~ok])
                    if lb > 0:
                        summary += f'; scene FRP {tot:.0f} MW retrieved + >= {lb:.0f} MW in saturated pixels'
        else:
            self.tiles['profile'][0].configure(text='not run')
            self.tiles['quality'][0].configure(text='-')
        self._populate_preview_methods(results_df)

        pml2 = attrs.get('profile_ml_alt')
        p2 = pml2.get('df') if isinstance(pml2, dict) else None
        if p2 is not None and len(p2):
            ok2 = (p2['fit_quality'].values == 'ok') & np.isfinite(p2['T_K'].values)
            fin2 = ok2 & ~(p2['T_lower_unbounded'].values | p2['T_upper_unbounded'].values)
            med2 = float(np.nanmedian(p2['T_K'].values[ok2])) if ok2.any() else float('nan')
            w2 = float(np.nanmedian((p2['T_ci_high'] - p2['T_ci_low']).values[fin2])) if fin2.any() else float('nan')
            self._show_v1_tile(True)
            self.tiles['legacy_v1'][0].configure(text=f'{med2:.0f} K' if np.isfinite(med2) else '-')
            self.tiles['legacy_v1'][1].configure(
                text=(f'median 95% interval ±{w2 / 2:.0f} K; {int(ok2.sum())} adequate' if np.isfinite(w2)
                      else 'no finite interval'))
            summary += f'; legacy v1 median {med2:.0f} K'
        else:
            self._show_v1_tile(False)

        self._report_path = attrs.get('report_path')
        if self._report_path and os.path.exists(self._report_path):
            self.open_report_btn.configure(state='normal')
            self._log(f'\nPDF report: {self._report_path}\n')

        self._set_status('done', 'Finished')
        self._log(f'\nDone. {summary}.\n')
        if figure_paths:
            self._log('Figures written:\n' + '\n'.join(f'  {p}' for p in figure_paths) + '\n')
        self.open_folder_btn.configure(state='normal')
        self.save_plot_btn.configure(state='normal')
        self._render_selected_preview()

    def _on_error(self, message):
        self._set_running(False)
        self._set_status('error', 'Failed - see the log')
        self.stage_label.configure(text='Failed')
        self._log('\n--- ERROR ---\n' + message + '\n')
        messagebox.showerror('Pipeline failed', message.splitlines()[-1] if message else 'Unknown error')

    def _on_cancelled(self, message):
        self._set_running(False)
        self._set_status('cancelled', 'Cancelled')
        self.stage_label.configure(text='Cancelled')
        self._log(f'\n--- CANCELLED ---\n{message}\n')

    def _set_running(self, running):
        self.run_button.config(state='disabled' if running else 'normal')
        self.cancel_button.config(state='normal' if running else 'disabled')
        if not running:
            self._cancel_event = None
        if running:
            self.progress.configure(mode='determinate', maximum=100, value=0)
            self.percent_label.configure(text='')
            self.stage_label.configure(text='Starting')
            self._set_status('running', 'Running')
        else:
            self.progress.configure(mode='determinate', value=0)
            self.percent_label.configure(text='')

    def _set_status(self, kind, text):
        colour = {'idle': self.t['muted'], 'running': self.t['info'], 'done': self.t['ok'],
                  'error': self.t['err'], 'cancelled': self.t['warn']}[kind]
        glyph = {'idle': '●', 'running': '◔', 'done': '✓', 'error': '✕',
                 'cancelled': '■'}[kind]
        self.status_label.configure(text=f'{glyph}  {text}', foreground=colour)

    # ------------------------------------------------------------ log
    def _configure_log_tags(self):
        t = self.t
        mono = tkfont.Font(family=MONO, size=10)
        bold = tkfont.Font(family=MONO, size=10, weight='bold')
        self.log_text.configure(font=mono, spacing1=1, spacing3=2)
        self.log_text.tag_configure('error', foreground=t['log_err'], font=bold)
        self.log_text.tag_configure('warn', foreground=t['log_warn'])
        self.log_text.tag_configure('success', foreground=t['log_ok'])
        self.log_text.tag_configure('stage', foreground=t['log_stage'], font=bold, spacing1=8)

    def _clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.config(state='disabled')

    def _log(self, text):
        self.log_text.config(state='normal')
        for line in text.splitlines(keepends=True):
            low = line.lower()
            stripped = line.strip()
            tag = None
            if stripped.startswith('[') and '/' in stripped[:6] and ']' in stripped[:8]:
                tag = 'stage'
            elif 'traceback' in low or 'error' in low or 'failed' in low:
                tag = 'error'
            elif 'warn' in low:
                tag = 'warn'
            elif low.lstrip().startswith(('done', 'saved')):
                tag = 'success'
            self.log_text.insert('end', line, tag) if tag else self.log_text.insert('end', line)
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    # ------------------------------------------------------------ preview
    def _panel_bg_color(self):
        return self.t['plot_bg']

    def _refresh_preview_background(self):
        bg = self._panel_bg_color()
        self.figure.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.cax.set_facecolor(bg)
        try:
            self.canvas.get_tk_widget().configure(bg=bg)
        except Exception:
            pass
        if self._last_result is not None:
            self._render_selected_preview()
        else:
            self._show_preview_placeholder('Run the pipeline to see\nthe temperature map here.')

    def _draw_colorbar(self, mappable, **kw):
        """Colorbar on the fixed self.cax, so the map axes never change size."""
        self.cax.clear()
        self.cax.set_visible(True)
        self._colorbar = self.figure.colorbar(mappable, cax=self.cax, label='Temperature (K)', **kw)
        self._colorbar.ax.set_facecolor(self._panel_bg_color())
        self._colorbar.ax.yaxis.label.set_color(self.t['ink'])
        self._colorbar.ax.tick_params(colors=self.t['ink'])

    def _clear_colorbar(self):
        self.cax.clear()
        self.cax.set_visible(False)
        self._colorbar = None

    def _zoom_to_fire(self, ind_x, ind_y, shape, lon=None, lat=None):
        """Open the map on the fire window the comparison view and the PDF
        report use (scene_report._window: the dense core of the detections, at
        least 40 px a side), not the whole scene. The toolbar's Home button
        returns here; scroll or the toolbar zooms out."""
        import scene_report as sr
        import pandas as pd
        core = {'ind_x': pd.Series(ind_x), 'ind_y': pd.Series(ind_y)}
        r0, r1, c0, c1 = sr._window({'fire': core}, shape[0], shape[1])
        if lon is None:
            self.ax.set_xlim(c0 - 0.5, c1 - 0.5)
            self.ax.set_ylim(r1 - 0.5, r0 - 0.5)
        else:
            wlon, wlat = lon[r0:r1, c0:c1], lat[r0:r1, c0:c1]
            self.ax.set_xlim(float(np.nanmin(wlon)), float(np.nanmax(wlon)))
            self.ax.set_ylim(float(np.nanmin(wlat)), float(np.nanmax(wlat)))

    def _on_map_scroll(self, event):
        """Mouse wheel zooms the map about the cursor."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        f = 1 / 1.25 if event.button == 'up' else 1.25
        if self._toolbar._nav_stack() is None:
            self._toolbar.push_current()  # so Home can undo a wheel zoom
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim(cx + (x0 - cx) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy + (y0 - cy) * f, cy + (y1 - cy) * f)
        self.canvas.draw_idle()

    def _finish_map(self):
        # Reset the toolbar's view history so Home returns to this map's initial view.
        try:
            self._toolbar.update()
        except Exception:
            pass
        self._set_map_tools(True)

    def _set_map_tools(self, enabled):
        for b in self._map_tools.values():
            b.state(['!disabled'] if enabled else ['disabled'])
        if not enabled:
            mode = getattr(self._toolbar.mode, 'name', 'NONE')
            if mode in ('PAN', 'ZOOM'):
                getattr(self._toolbar, mode.lower())()   # toggles the mode off
            self._map_coords.set('')
        self._paint_map_tools()

    def _map_mode(self, mode):
        """Pan / zoom-to-rectangle toggles of the (hidden) matplotlib toolbar."""
        getattr(self._toolbar, mode)()
        self._paint_map_tools()

    def _paint_map_tools(self):
        mode = getattr(self._toolbar.mode, 'name', 'NONE')
        for key in ('pan', 'zoom'):
            self._map_tools[key].configure(
                style='ToolOn.TButton' if mode == key.upper() else 'Tool.TButton')

    def _show_compare_row(self, show, label_a='', label_b=''):
        if show:
            self._compare_a_label.configure(text=f'◀  {label_a}')
            self._compare_b_label.configure(text=f'{label_b}  ▶')
            if not self._compare_row.winfo_manager():
                self._compare_row.pack(fill='x', pady=(8, 0), after=self._preview_pickers)
        elif self._compare_row.winfo_manager():
            self._compare_row.pack_forget()

    def _show_preview_placeholder(self, text, title='No map yet'):
        """Empty state: a dashed drop-zone frame with a heading, the hint and
        the inferno ramp the map will use, centred on the whole figure (the
        map axes leave room for a colorbar and would sit off-centre)."""
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.collections import LineCollection
        from matplotlib.transforms import offset_copy
        self._clear_colorbar()
        self.ax.clear()
        self.ax.set_facecolor(self._panel_bg_color())
        fig_tr = self.figure.transFigure
        self.ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92, transform=fig_tr, clip_on=False,
                                         boxstyle='round,pad=0,rounding_size=0.02', fill=False,
                                         ec=self.t['field_line'], lw=1.2, ls=(0, (5, 4))))
        # Vertical spacing in points, so the stack never overlaps on a short canvas.
        n = 64
        xs = np.linspace(0.38, 0.62, n + 1)
        ramp = LineCollection([[(xs[i], 0.5), (xs[i + 1], 0.5)] for i in range(n)],
                              colors=matplotlib.colormaps['inferno'](np.linspace(0, 1, n)),
                              linewidths=5, capstyle='butt', clip_on=False,
                              transform=offset_copy(fig_tr, fig=self.figure, y=30, units='points'))
        self.ax.add_collection(ramp)
        self.ax.annotate(title, (0.5, 0.5), xycoords='figure fraction', xytext=(0, 4),
                         textcoords='offset points', ha='center', va='bottom',
                         color=self.t['ink'], fontsize=12, fontweight='bold')
        self.ax.annotate(text, (0.5, 0.5), xycoords='figure fraction', xytext=(0, -4),
                         textcoords='offset points', ha='center', va='top',
                         color=self.t['muted'], fontsize=9.5)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_axis_off()
        self._show_compare_row(False)
        if hasattr(self, '_map_tools'):
            self._set_map_tools(False)
        self.canvas.draw()

    # ------------------------------------------------------------ basemap
    def _basemap_year(self):
        """Mosaic year the Background list asks for; None for the EnMAP composite."""
        choice = self.background.get()
        if choice == self.BG_ENMAP:
            return None
        if choice == self.BG_S2_AUTO:
            return basemap.layer_for_acquisition(self._acq_dt)
        return int(choice.rsplit(' ', 1)[-1])

    def _current_basemap(self):
        """The fetched mosaic for the chosen year, or None while it loads, if
        it failed, or when the EnMAP composite is chosen. A missing mosaic is
        fetched off the UI thread; the map is redrawn when it arrives (and
        shows the EnMAP composite until then)."""
        year = self._basemap_year()
        scene = self._last_result[1] if self._last_result is not None else None
        if year is None or scene is None or scene.get('lon') is None:
            self._note_background()
            return None
        if year in self._basemaps:
            self._note_background()
            return self._basemaps[year]
        if year not in self._basemap_failed and year not in self._basemap_pending:
            self._basemap_pending.add(year)
            token = id(self._last_result)
            bounds = basemap.scene_bounds(scene['lon'], scene['lat'])
            # kept with the scene's other downloads
            cache = os.path.join(self._scene_dir, 'downloads') if self._scene_dir else None

            def work():
                try:
                    self._queue.put(('BASEMAP', (token, year, basemap.fetch_basemap(bounds, year, cache))))
                except Exception as exc:
                    self._queue.put(('BASEMAP', (token, year, exc)))
            threading.Thread(target=work, daemon=True).start()
        self._note_background()
        return None

    def _on_basemap(self, token, year, result):
        if self._last_result is None or token != id(self._last_result):
            return                      # a mosaic for an earlier run
        self._basemap_pending.discard(year)
        if isinstance(result, Exception):
            self._basemap_failed[year] = f'{result.__class__.__name__}: {result}'
            self._log(f'\nSentinel-2 cloudless {year} basemap not available ({self._basemap_failed[year]}); '
                      'the map shows the EnMAP composite.\n')
        else:
            self._basemaps[year] = result
            self._log(f'\nBasemap: {result["attribution"]} (zoom {result["zoom"]}).\n')
        if year == self._basemap_year():
            self._render_selected_preview()

    def _note_background(self):
        """Status beside the Background list: loading, failed, or the mosaic's
        licence and whether the fire pixels sit on co-registered positions."""
        year = self._basemap_year()
        scene = self._last_result[1] if self._last_result is not None else None
        if year is None or scene is None:
            text = ''
        elif year in self._basemap_pending:
            text = f'Loading the {year} mosaic…'
        elif year in self._basemap_failed:
            text = f'{year} mosaic unavailable (offline?); EnMAP composite shown'
        else:
            geo = ('positions co-registered (AROSICS)' if 'lat_nominal' in scene
                   else 'nominal L1B positions, may be offset')
            text = f'{year} mosaic, {basemap.licence(year)}; {geo}'
        self._background_note.configure(text=text)

    def _draw_attribution(self, bm):
        """EOX credit on the map itself, so a saved PNG carries it. Short enough
        for a narrow map; the full credit line is in the log and basemap.attribution()."""
        y = bm['year']
        self.ax.text(0.99, 0.01, f'Sentinel-2 cloudless {y} © EOX\n'
                     f'modified Copernicus Sentinel data {y}',
                     transform=self.ax.transAxes, ha='right', va='bottom', fontsize=5.5,
                     color='#ffffff', zorder=5, linespacing=1.2,
                     bbox=dict(boxstyle='square,pad=0.3', fc='#000000', ec='none', alpha=0.5))

    def _show_v1_tile(self, show):
        """The legacy-v1 tile exists only for runs that computed v1; its grid
        column collapses when it is hidden so the other tiles close up."""
        frame = self._tile_frames['legacy_v1']
        col = int(frame.grid_info().get('column', 2)) if frame.winfo_manager() else 2
        if show:
            self._tiles_parent.columnconfigure(col, weight=1, uniform='tile')
            frame.grid()
        else:
            frame.grid_remove()
            self._tiles_parent.columnconfigure(col, weight=0, uniform='')

    def _populate_preview_methods(self, results_df):
        """Fill both preview lists with the methods that actually ran in this
        run, primary first. Every method's usable pixels follow the report's
        statistics policy (scene_report._normalise): converged / adequate
        fit, not at the search bound, interval not open at both ends."""
        import scene_report as sr
        attrs = getattr(results_df, 'attrs', {}) or {}
        variant = attrs.get('profile_ml_variant', 'v2')
        key_primary = 'ProfileMLv2' if variant == 'v2' else 'ProfileML'
        key_alt = 'ProfileML' if variant == 'v2' else 'ProfileMLv2'
        frames = []
        for key, res, tag in ((key_primary, attrs.get('profile_ml'), ' - primary'),
                              (key_alt, attrs.get('profile_ml_alt'), ' - legacy' if variant == 'v2' else '')):
            df = res.get('df') if isinstance(res, dict) else None
            if df is not None and len(df):
                frames.append((sr.LABELS[key] + tag, key, df))
        if len(results_df) and 'mean_temp_K' in results_df:
            frames.append((sr.LABELS['Amici2022'], 'Amici2022', results_df))
        for key in ('Dennison2006', 'Dozier1981', 'SmokeSTCR'):
            df = (attrs.get('method_results') or {}).get(key)
            if df is not None and len(df):
                frames.append((sr.LABELS[key], key, df))
        self._preview_methods = {}
        for label, key, df in frames:
            try:
                self._preview_methods[label] = sr._normalise(key, df)
            except Exception as exc:
                self._log(f'\nPreview: could not read the {label} results ({exc}).\n')
        labels = list(self._preview_methods)
        self._compare_state = None
        self.preview_combo.configure(values=labels, state='readonly' if labels else 'disabled')
        self.preview_source.set(labels[0] if labels else '')
        self.compare_combo.configure(values=[self.NO_COMPARE] + labels,
                                     state='readonly' if len(labels) > 1 else 'disabled')
        self.compare_source.set(self.NO_COMPARE)
        self.compare_scale.state(['disabled'])

    def _render_selected_preview(self):
        if self._last_result is None:
            return
        _, scene, temp_map, _ = self._last_result
        src = self.preview_source.get()
        other = self.compare_source.get()
        m = self._preview_methods.get(src)
        if m is None:
            self._show_preview_placeholder('No retrieval to display.')
            return
        if other in self._preview_methods and other != src:
            self.compare_scale.state(['!disabled'])
            self._show_compare_row(True, src, other)
            self._render_compare(scene, src, other)
            return
        self.compare_scale.state(['disabled'])
        self._show_compare_row(False)
        self._compare_state = None
        tmap = np.full(temp_map.shape, np.nan, dtype='float32')
        v = m['valid'].values
        tmap[m['ind_x'].values[v].astype(int), m['ind_y'].values[v].astype(int)] = m['T'].values[v]
        self._render_preview(scene, tmap, f'{src} · {int(v.sum())}/{len(v)} retrieved')

    def _compare_rgba(self, m, win, grey, norm):
        """RGBA image of one method on the window: grey scene, retrieved
        temperatures in the shared colour scale, detected pixels without a
        usable retrieval of THIS method in blue (as in the report maps)."""
        r0, r1, c0, c1 = win
        img = grey if grey.ndim == 3 else np.repeat(grey[..., None], 3, axis=2)   # basemap is RGB
        img = np.concatenate([img, np.ones(img.shape[:2] + (1,))], axis=2)
        xi = m['ind_x'].values.astype(int) - r0
        yi = m['ind_y'].values.astype(int) - c0
        inside = (xi >= 0) & (xi < r1 - r0) & (yi >= 0) & (yi < c1 - c0)
        v = m['valid'].values
        nv = inside & ~v
        img[xi[nv], yi[nv]] = (0.12, 0.47, 0.71, 1.0)
        ok = inside & v
        img[xi[ok], yi[ok]] = matplotlib.colormaps['inferno'](norm(m['T'].values[ok].astype(float)))
        return img

    def _render_compare(self, scene, label_a, label_b):
        """Comparison slider: method A left of the divider, method B right,
        on the same 30 m pixel grid (the report's fire window) and one colour
        scale (lowest 2nd percentile to the median of the 98th percentiles)."""
        import scene_report as sr
        from matplotlib.colors import Normalize
        a, b = self._preview_methods[label_a], self._preview_methods[label_b]
        n_rows, n_cols = self._last_result[2].shape[:2]
        win = sr._window({label_a: a}, n_rows, n_cols)
        r0, r1, c0, c1 = win
        bm = self._current_basemap()
        scene_ll = self._last_result[1]
        if bm is not None:
            # the mosaic sampled at each 30 m pixel's lon/lat, so both halves share one grid
            grey = basemap.sample(bm, scene_ll['lon'][r0:r1, c0:c1], scene_ll['lat'][r0:r1, c0:c1])
            grey = np.where(np.isfinite(grey), grey, 0.85)
        elif self._last_rgb_composite is not None:
            rgb = self._last_rgb_composite[r0:r1, c0:c1, :3].astype(float)
            grey = 0.15 + 0.6 * (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        else:
            grey = np.full((r1 - r0, c1 - c0), 0.85)
        vals = [m['T'].values[m['valid'].values].astype(float) for m in (a, b)]
        vals = [x for x in vals if len(x)]
        vmin = min(np.percentile(x, 2) for x in vals) if vals else 500.0
        vmax = float(np.median([np.percentile(x, 98) for x in vals])) if vals else 1400.0
        norm = Normalize(vmin=float(vmin), vmax=max(float(vmax), float(vmin) + 1.0))
        self._compare_state = {'A': self._compare_rgba(a, win, grey, norm),
                               'B': self._compare_rgba(b, win, grey, norm), 'win': win}
        bg = self._panel_bg_color()
        self.ax.clear()
        self.ax.set_facecolor(bg)
        ext = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
        self._compare_state['im'] = self.ax.imshow(self._compare_state['A'], extent=ext,
                                                   interpolation='nearest')
        self._compare_state['line'] = self.ax.axvline(c0, color='white', lw=1.6)
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(norm=norm, cmap='inferno')
        self._draw_colorbar(sm, extend='both')
        n_out = int((~((a['ind_x'] >= r0) & (a['ind_x'] < r1) & (a['ind_y'] >= c0) & (a['ind_y'] < c1))).sum())
        va, vb = a['valid'].values, b['valid'].values
        # method names and retrieved/detected counts label the slider ends (a two-line
        # axes title was clipped once the slider row took its height from the canvas)
        self._show_compare_row(True, f'{label_a}  ({int(va.sum())}/{len(va)})',
                               f'{label_b}  ({int(vb.sum())}/{len(vb)})')
        self.ax.text(0.5, -0.02, 'blue: detected, no usable retrieval'
                     + (f'\n{n_out} detections outside this window' if n_out else ''),
                     transform=self.ax.transAxes, ha='center', va='top', fontsize=7,
                     color=self.t['muted'])
        if bm is not None:
            self._draw_attribution(bm)
        self.ax.set_axis_off()
        self._finish_map()
        self._update_compare_split()

    def _update_compare_split(self):
        st = self._compare_state
        if not st or 'im' not in st:
            return
        r0, r1, c0, c1 = st['win']
        split = int(round(float(self.compare_split.get()) * (c1 - c0)))
        img = st['B'].copy()
        img[:, :split] = st['A'][:, :split]
        st['im'].set_data(img)
        st['line'].set_xdata([c0 - 0.5 + split] * 2)
        self.canvas.draw_idle()

    def _render_preview(self, scene, temp_map, title):
        bg = self._panel_bg_color()
        self.figure.set_facecolor(bg)
        self._clear_colorbar()
        self.ax.clear()
        self.ax.set_facecolor(bg)
        finite = np.isfinite(temp_map)
        if finite.any():
            ind_x, ind_y = np.where(finite)
            values = temp_map[ind_x, ind_y]
            lat, lon = scene['lat'], scene['lon']
            bm = self._current_basemap()
            if bm is not None:
                # Sentinel-2 cloudless mosaic on its lon/lat grid, fire markers on top
                self.ax.imshow(bm['img'], extent=bm['extent'], origin='upper', interpolation='bilinear',
                               zorder=0)
                pcm = pipeline._scatter_over_composite(
                    self.ax, lon, lat, None, ind_x, ind_y, values,
                    'inferno', float(values.min()), float(values.max()))
                self.ax.set_aspect(1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3))
                self._zoom_to_fire(ind_x, ind_y, temp_map.shape, lon, lat)
                self._draw_attribution(bm)
            elif self._last_rgb_composite is not None:
                pcm = pipeline._scatter_over_composite(
                    self.ax, lon, lat, self._last_rgb_composite, ind_x, ind_y, values,
                    'inferno', float(values.min()), float(values.max()))
                self.ax.set_aspect(1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3))
                self._zoom_to_fire(ind_x, ind_y, temp_map.shape, lon, lat)
            else:
                cmap = matplotlib.colormaps['inferno'].copy()
                cmap.set_bad(color=bg)
                pcm = self.ax.imshow(temp_map, cmap=cmap, interpolation='nearest')
                self._zoom_to_fire(ind_x, ind_y, temp_map.shape)
            self._draw_colorbar(pcm)
            self.ax.set_title(title, color=self.t['ink'], fontsize=9, loc='left')
        else:
            self.ax.text(0.5, 0.5, 'No finite temperature values to display',
                         ha='center', va='center', color=self.t['muted'])
        self.ax.set_axis_off()
        self._finish_map()
        self.canvas.draw()

    # ------------------------------------------------------------ post-run actions
    def _open_results_folder(self):
        # this run's scene folder (results + downloads/); the parent before any run
        path = self._scene_dir if self._scene_dir and os.path.isdir(self._scene_dir) else self.results_dir.get()
        if path and os.path.isdir(path):
            if sys.platform.startswith('win'):
                os.startfile(path)
            elif sys.platform == 'darwin':
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        else:
            messagebox.showwarning('Folder not found', f'{path} does not exist yet.')

    def _open_report(self):
        path = self._report_path
        if not path or not os.path.exists(path):
            messagebox.showwarning('No report', 'No PDF report was written for this run.')
            return
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def _on_build_report(self):
        """Typeset the PDF report of a scene processed earlier, from its
        results folder alone (scene_report.build_scene_report)."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            messagebox.showinfo('Busy', 'Wait for the current run to finish first.')
            return
        start = self.results_dir.get() if os.path.isdir(self.results_dir.get()) else os.getcwd()
        folder = filedialog.askdirectory(title="Select a scene's results folder (results_temperature/<scene>)",
                                         initialdir=start)
        if not folder:
            return
        self._clear_log()
        self._show_page('run')
        self._set_status('running', 'Building the PDF report')
        self._log(f'Building the PDF report for {folder} ...\n')

        def work():
            try:
                import scene_report
                path = scene_report.build_scene_report(folder)
                self._queue.put(('REPORT', path))
            except Exception:
                self._queue.put(('ERROR', traceback.format_exc()))

        self._worker_thread = threading.Thread(target=_capturing(self._queue, work), daemon=True)
        self._worker_thread.start()

    def _on_report_built(self, path):
        self._report_path = path
        self.open_report_btn.configure(state='normal')
        self._set_status('done', 'Report written')
        self._log(f'\nPDF report: {path}\n')

    def _save_preview(self):
        if self._last_result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG image', '*.png')])
        if path:
            self.figure.savefig(path, dpi=200, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Preview saved to {path}')

    def _show_about(self):
        messagebox.showinfo(
            'About',
            'EnMAP Wildfire Temperature\n\n'
            'Sub-pixel fire temperature from EnMAP L1B/L1C at-sensor radiance.\n\n'
            'Primary method: this work (2026), a profile-likelihood inversion of the '
            'Planck-plus-background mixture with a scene-estimated fire-term transmittance, '
            'L1B quality masks and 95% intervals; variant 2 iterates the error-model '
            'reweighting to its fixed point and estimates the scene-level quantities from '
            'adequately fitted pixels only.\n'
            'Reference method: Amici et al. (2022), the multi-trial Planck-mixture fit of the '
            'PRISMA pipeline this software descends from, run in its published configuration '
            '(1400-1900 nm, 20 background spectra, 15 trials).\n'
            'Also available: Dozier (1981), Dennison et al. (2006).\n\n'
            f'{self._gdal_msg}')


if __name__ == '__main__':
    app = EnMAPFireGUI()
    app.mainloop()
