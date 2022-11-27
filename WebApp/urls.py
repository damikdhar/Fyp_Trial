from django.urls import path,include
from . import views
from . import views
urlpatterns=[
    path('',views.appindex,name='app-index'),
    path('result/',views.result,name='result'),
    path('trial',views.trial,name='trial'),
    path('dia/',views.dia,name='dia'),
    path('heart/',views.heart,name='heart'),
    path('hresult/',views.hresult,name='hresult'),
   
   
]
