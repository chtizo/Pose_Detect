from django import forms  

class VideoForm(forms.Form): 
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'upload', 'hidden': 'true'})
    file = forms.FileField()
    file.widget.attrs.update({'hidden': 'true'})

class Button1(forms.Form): 
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'button1', 'hidden': 'true'})

class Button2(forms.Form): 
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'button2', 'hidden': 'true'})

class Button3(forms.Form): 
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'button3', 'hidden': 'true'})

class AnalyseConfirm(forms.Form):
    ins = forms.CharField()
    ins.widget.attrs.update({'name': 'ins', 'value': 'analyse', 'hidden': 'true'})
    test = forms.CharField() 