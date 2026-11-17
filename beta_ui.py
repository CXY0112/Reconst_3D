"""Demo layout over the same stage controllers; algorithms and saved formats are shared."""
def _hide_labels(root,variables):
    for widget in root.winfo_children():
        if 'textvariable' in widget.keys() and str(widget.cget('textvariable')) in variables and widget.winfo_manager()=='pack':
            widget.pack_forget()
        _hide_labels(widget,variables)


def apply_beta(view,stage):
    view.beta=True
    if stage=='acquisition':
        view.preprocess_button.pack_forget()
        if hasattr(view,'rate_row'):view.rate_row.pack_forget()
    if stage=='preprocessing':
        # Remove both parameter rows, including units/axes; retain their default state.
        view.fields['noise_mm'].master.pack_forget()
        view.fields['kalman_floor_mm'].master.pack_forget()
        view.view.pack_forget()
    if stage=='reconstruction':
        view.method.set('Poisson');view.method_box.pack_forget()
    for name in ('latest_button',):
        if hasattr(view,name):getattr(view,name).pack_forget()
    hidden={str(getattr(view,name)) for name in ('summary','source_text','method_text','performance') if hasattr(view,name)}
    for name in ('before','after','map_panel','live_panel','left_mesh','left_cloud'):
        panel=getattr(view,name,None)
        if hasattr(panel,'info'):hidden.add(str(panel.info))
    _hide_labels(view.root,hidden)
    for widget in view.root.winfo_children():
        if widget.winfo_class()=='TLabel' and widget.cget('text') and not widget.cget('textvariable'):
            widget.pack_forget()
    def simplify(widget):
        for child in widget.winfo_children():
            if child.winfo_class()=='TCheckbutton' and child.winfo_manager()=='pack':child.pack_forget()
            if child.winfo_class()=='TCombobox' and hasattr(view,'mode') and str(child.cget('textvariable'))==str(view.mode):child.pack_forget()
            simplify(child)
    simplify(view.root)
