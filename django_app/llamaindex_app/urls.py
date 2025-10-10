from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('search/', views.search, name='search'),
    path('upload/', views.upload, name='upload'),
    path('admin_panel/', views.admin_panel, name='admin_panel'),
    path('models/', views.models_view, name='models'),
    path('pull_model/', views.pull_model, name='pull_model'),
    path('delete_model/', views.delete_model, name='delete_model'),
    path('change-model/', views.change_model, name='change_model'),
    path('update-config/', views.update_config, name='update_config'),
    path('get-stats/', views.get_stats, name='get_stats'),
    path('change-kb/', views.change_kb, name='change_kb'),
    path('get-status/', views.get_status, name='get_status'),
    path('delete-db/', views.delete_db, name='delete_db'),
    path('delete_file/', views.delete_file, name='delete_file'),
    path('save-history/', views.save_history, name='save_history'),
    path('load-history/', views.load_history, name='load_history'),
    path('.well-known/appspecific/com.chrome.devtools.json', lambda r: HttpResponse(status=204)),
]