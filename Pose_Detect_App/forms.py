from django import forms  

class VideoForm(forms.Form): 
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'upload', 'hidden': 'true'})
    file = forms.FileField()
    file.widget.attrs.update({'hidden': 'true'})

class AnalyseConfirm(forms.Form):
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'analyse', 'hidden': 'true'})
    type = forms.CharField()
    type.widget.attrs.update({'name': 'type', 'value': '', 'hidden': 'true'})