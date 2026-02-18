"""
Sistema de Alertas Simplificado con Telegram
=============================================
Solo introduce tus compras y el sistema hace el resto.

Configuración inicial (solo una vez):
    python alertas_telegram.py --setup

Añadir una compra:
    python alertas_telegram.py --comprar AAPL 150.50 10

Vigilancia continua:
    python alertas_telegram.py --vigilar
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime

# --------------------------------------------------
# ARCHIVO DE CONFIGURACIÓN
# --------------------------------------------------
CONFIG_FILE = "config_alertas.json"
CARTERA_FILE = "mi_cartera.json"

# Niveles automáticos (porcentajes respecto al precio de compra)
NIVELES = {
    "stop_loss": -0.10,        # -10% → Vender para limitar pérdidas
    "alerta_baja": -0.05,      # -5%  → Aviso de caída
    "alerta_alta": +0.10,      # +10% → Aviso de subida
    "take_profit": +0.20,      # +20% → Vender para asegurar ganancias
    "take_profit_2": +0.30,    # +30% → Segundo objetivo
}


# --------------------------------------------------
# CONFIGURACIÓN DE TELEGRAM
# --------------------------------------------------

def cargar_config():
    """Carga la configuración de Telegram."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def guardar_config(config):
    """Guarda la configuración."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def setup_telegram():
    """Configura Telegram paso a paso."""
    print("\n" + "="*50)
    print("🔧 CONFIGURACIÓN DE TELEGRAM")
    print("="*50)
    
    print("""
PASO 1: Crear tu bot
--------------------
1. Abre Telegram y busca: @BotFather
2. Envíale: /newbot
3. Ponle un nombre (ej: "Mis Alertas Trading")
4. Ponle un username (ej: "pedro_alertas_bot")
5. BotFather te dará un TOKEN como este:
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
""")
    
    token = input("Pega aquí tu TOKEN: ").strip()
    
    print("""
PASO 2: Obtener tu Chat ID
--------------------------
1. Busca tu bot en Telegram (el que acabas de crear)
2. Envíale cualquier mensaje (ej: "hola")
3. Pulsa Enter aquí y te diré tu Chat ID...
""")
    
    input("Pulsa Enter después de enviar un mensaje a tu bot...")
    
    # Obtener chat_id automáticamente
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        response = requests.get(url)
        data = response.json()
        
        if data["result"]:
            chat_id = str(data["result"][-1]["message"]["chat"]["id"])
            username = data["result"][-1]["message"]["chat"].get("username", "")
            print(f"\n✅ ¡Encontrado! Tu Chat ID es: {chat_id}")
            if username:
                print(f"   Usuario: @{username}")
        else:
            print("\n❌ No encontré mensajes. Asegúrate de enviar algo a tu bot.")
            chat_id = input("Introduce tu Chat ID manualmente: ").strip()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        chat_id = input("Introduce tu Chat ID manualmente: ").strip()
    
    # Guardar configuración
    config = {
        "telegram_token": token,
        "telegram_chat_id": chat_id,
        "intervalo_minutos": 5
    }
    guardar_config(config)
    
    # Enviar mensaje de prueba
    print("\n📤 Enviando mensaje de prueba...")
    if enviar_telegram("🎉 ¡Bot configurado correctamente!\n\nRecibirás alertas de tu cartera aquí."):
        print("✅ ¡Configuración completada! Revisa Telegram.")
    else:
        print("❌ Error al enviar. Verifica el token y chat_id.")
    
    return config


def enviar_telegram(mensaje):
    """Envía mensaje por Telegram."""
    config = cargar_config()
    
    if not config.get("telegram_token") or not config.get("telegram_chat_id"):
        print("⚠️ Telegram no configurado. Ejecuta: python alertas_telegram.py --setup")
        return False
    
    url = f"https://api.telegram.org/bot{config['telegram_token']}/sendMessage"
    payload = {
        "chat_id": config["telegram_chat_id"],
        "text": mensaje,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json().get("ok", False)
    except Exception as e:
        print(f"Error enviando Telegram: {e}")
        return False


# --------------------------------------------------
# GESTIÓN DE CARTERA
# --------------------------------------------------

def cargar_cartera():
    """Carga la cartera guardada."""
    if os.path.exists(CARTERA_FILE):
        with open(CARTERA_FILE, "r") as f:
            return json.load(f)
    return {"posiciones": [], "alertas_enviadas": {}}


def guardar_cartera(cartera):
    """Guarda la cartera."""
    with open(CARTERA_FILE, "w") as f:
        json.dump(cartera, f, indent=2, default=str)


def añadir_compra(ticker, precio_compra, cantidad):
    """Añade una compra a la cartera."""
    ticker = ticker.upper()
    cartera = cargar_cartera()
    
    # Calcular niveles automáticamente
    niveles = {
        "stop_loss": round(precio_compra * (1 + NIVELES["stop_loss"]), 2),
        "alerta_baja": round(precio_compra * (1 + NIVELES["alerta_baja"]), 2),
        "alerta_alta": round(precio_compra * (1 + NIVELES["alerta_alta"]), 2),
        "take_profit": round(precio_compra * (1 + NIVELES["take_profit"]), 2),
        "take_profit_2": round(precio_compra * (1 + NIVELES["take_profit_2"]), 2),
    }
    
    posicion = {
        "ticker": ticker,
        "precio_compra": precio_compra,
        "cantidad": cantidad,
        "fecha_compra": datetime.now().isoformat(),
        "inversion": round(precio_compra * cantidad, 2),
        "niveles": niveles
    }
    
    # Buscar si ya existe el ticker
    existente = None
    for i, p in enumerate(cartera["posiciones"]):
        if p["ticker"] == ticker:
            existente = i
            break
    
    if existente is not None:
        # Actualizar posición existente (promedio)
        pos_actual = cartera["posiciones"][existente]
        cantidad_total = pos_actual["cantidad"] + cantidad
        precio_medio = (pos_actual["precio_compra"] * pos_actual["cantidad"] + 
                       precio_compra * cantidad) / cantidad_total
        
        posicion["cantidad"] = cantidad_total
        posicion["precio_compra"] = round(precio_medio, 2)
        posicion["inversion"] = round(precio_medio * cantidad_total, 2)
        
        # Recalcular niveles con nuevo precio medio
        posicion["niveles"] = {
            "stop_loss": round(precio_medio * (1 + NIVELES["stop_loss"]), 2),
            "alerta_baja": round(precio_medio * (1 + NIVELES["alerta_baja"]), 2),
            "alerta_alta": round(precio_medio * (1 + NIVELES["alerta_alta"]), 2),
            "take_profit": round(precio_medio * (1 + NIVELES["take_profit"]), 2),
            "take_profit_2": round(precio_medio * (1 + NIVELES["take_profit_2"]), 2),
        }
        
        cartera["posiciones"][existente] = posicion
        print(f"✅ Actualizada posición en {ticker}")
        print(f"   Cantidad total: {cantidad_total} acciones")
        print(f"   Precio medio: ${precio_medio:.2f}")
    else:
        cartera["posiciones"].append(posicion)
        print(f"✅ Añadida compra de {ticker}")
    
    # Resetear alertas enviadas para este ticker
    cartera["alertas_enviadas"][ticker] = {}
    
    guardar_cartera(cartera)
    
    # Mostrar niveles
    print(f"\n📊 Niveles automáticos para {ticker}:")
    print(f"   🔴 Stop Loss (-10%):    ${posicion['niveles']['stop_loss']:.2f}")
    print(f"   🟠 Alerta baja (-5%):   ${posicion['niveles']['alerta_baja']:.2f}")
    print(f"   💰 Precio compra:       ${posicion['precio_compra']:.2f}")
    print(f"   🟢 Alerta alta (+10%):  ${posicion['niveles']['alerta_alta']:.2f}")
    print(f"   🎯 Take Profit (+20%):  ${posicion['niveles']['take_profit']:.2f}")
    print(f"   🎯 Take Profit 2 (+30%): ${posicion['niveles']['take_profit_2']:.2f}")
    
    # Notificar por Telegram
    mensaje = f"""
🛒 <b>NUEVA COMPRA REGISTRADA</b>

📈 <b>{ticker}</b>
💰 Precio: ${precio_compra:.2f}
📦 Cantidad: {cantidad} acciones
💵 Inversión: ${posicion['inversion']:.2f}

<b>Alertas configuradas:</b>
🔴 Stop Loss: ${posicion['niveles']['stop_loss']:.2f}
🎯 Take Profit: ${posicion['niveles']['take_profit']:.2f}
"""
    enviar_telegram(mensaje)
    
    return posicion


def eliminar_posicion(ticker):
    """Elimina una posición de la cartera."""
    ticker = ticker.upper()
    cartera = cargar_cartera()
    
    cartera["posiciones"] = [p for p in cartera["posiciones"] if p["ticker"] != ticker]
    if ticker in cartera["alertas_enviadas"]:
        del cartera["alertas_enviadas"][ticker]
    
    guardar_cartera(cartera)
    print(f"✅ Eliminada posición de {ticker}")


def ver_cartera():
    """Muestra el estado actual de la cartera."""
    cartera = cargar_cartera()
    
    if not cartera["posiciones"]:
        print("\n📭 Tu cartera está vacía")
        print("   Añade una compra con: python alertas_telegram.py --comprar AAPL 150.50 10")
        return
    
    print("\n" + "="*70)
    print("📊 MI CARTERA")
    print("="*70)
    
    total_invertido = 0
    total_actual = 0
    
    for pos in cartera["posiciones"]:
        ticker = pos["ticker"]
        
        # Obtener precio actual
        try:
            data = yf.download(ticker, period="1d", progress=False)
            precio_actual = data["Close"].iloc[-1]
        except:
            precio_actual = pos["precio_compra"]
        
        valor_actual = precio_actual * pos["cantidad"]
        ganancia = valor_actual - pos["inversion"]
        ganancia_pct = (ganancia / pos["inversion"]) * 100
        
        total_invertido += pos["inversion"]
        total_actual += valor_actual
        
        emoji = "🟢" if ganancia >= 0 else "🔴"
        
        print(f"\n{emoji} {ticker}")
        print(f"   Compra: ${pos['precio_compra']:.2f} × {pos['cantidad']} = ${pos['inversion']:.2f}")
        print(f"   Actual: ${precio_actual:.2f} × {pos['cantidad']} = ${valor_actual:.2f}")
        print(f"   P/L: ${ganancia:+.2f} ({ganancia_pct:+.2f}%)")
        print(f"   Stop Loss: ${pos['niveles']['stop_loss']:.2f} | Take Profit: ${pos['niveles']['take_profit']:.2f}")
    
    print("\n" + "-"*70)
    ganancia_total = total_actual - total_invertido
    ganancia_total_pct = (ganancia_total / total_invertido) * 100 if total_invertido > 0 else 0
    emoji_total = "🟢" if ganancia_total >= 0 else "🔴"
    
    print(f"{emoji_total} TOTAL: Invertido ${total_invertido:.2f} → Actual ${total_actual:.2f}")
    print(f"   Ganancia/Pérdida: ${ganancia_total:+.2f} ({ganancia_total_pct:+.2f}%)")


# --------------------------------------------------
# SISTEMA DE VIGILANCIA
# --------------------------------------------------

def comprobar_alertas():
    """Comprueba todas las posiciones y genera alertas."""
    cartera = cargar_cartera()
    
    if not cartera["posiciones"]:
        return []
    
    alertas = []
    
    for pos in cartera["posiciones"]:
        ticker = pos["ticker"]
        niveles = pos["niveles"]
        
        # Obtener precio actual
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if data.empty:
                continue
            precio_actual = data["Close"].iloc[-1]
        except Exception as e:
            print(f"Error obteniendo {ticker}: {e}")
            continue
        
        # Calcular ganancia/pérdida
        ganancia_pct = ((precio_actual - pos["precio_compra"]) / pos["precio_compra"]) * 100
        
        # Alertas ya enviadas hoy para este ticker
        enviadas = cartera["alertas_enviadas"].get(ticker, {})
        hoy = datetime.now().strftime("%Y-%m-%d")
        
        # Comprobar cada nivel
        if precio_actual <= niveles["stop_loss"] and enviadas.get("stop_loss") != hoy:
            alertas.append({
                "ticker": ticker,
                "tipo": "🔴 STOP LOSS",
                "precio_actual": precio_actual,
                "nivel": niveles["stop_loss"],
                "ganancia_pct": ganancia_pct,
                "mensaje": "¡VENDER! Límite de pérdidas alcanzado",
                "urgencia": "CRITICA"
            })
            enviadas["stop_loss"] = hoy
            
        elif precio_actual <= niveles["alerta_baja"] and enviadas.get("alerta_baja") != hoy:
            alertas.append({
                "ticker": ticker,
                "tipo": "🟠 ALERTA BAJA",
                "precio_actual": precio_actual,
                "nivel": niveles["alerta_baja"],
                "ganancia_pct": ganancia_pct,
                "mensaje": "Precio cayendo, vigilar de cerca",
                "urgencia": "MEDIA"
            })
            enviadas["alerta_baja"] = hoy
            
        elif precio_actual >= niveles["take_profit_2"] and enviadas.get("take_profit_2") != hoy:
            alertas.append({
                "ticker": ticker,
                "tipo": "🎯 TAKE PROFIT 2",
                "precio_actual": precio_actual,
                "nivel": niveles["take_profit_2"],
                "ganancia_pct": ganancia_pct,
                "mensaje": "¡+30%! Considera vender o ajustar stop",
                "urgencia": "ALTA"
            })
            enviadas["take_profit_2"] = hoy
            
        elif precio_actual >= niveles["take_profit"] and enviadas.get("take_profit") != hoy:
            alertas.append({
                "ticker": ticker,
                "tipo": "🎯 TAKE PROFIT",
                "precio_actual": precio_actual,
                "nivel": niveles["take_profit"],
                "ganancia_pct": ganancia_pct,
                "mensaje": "¡+20%! Objetivo alcanzado",
                "urgencia": "ALTA"
            })
            enviadas["take_profit"] = hoy
            
        elif precio_actual >= niveles["alerta_alta"] and enviadas.get("alerta_alta") != hoy:
            alertas.append({
                "ticker": ticker,
                "tipo": "🟢 ALERTA ALTA",
                "precio_actual": precio_actual,
                "nivel": niveles["alerta_alta"],
                "ganancia_pct": ganancia_pct,
                "mensaje": "¡Subiendo! +10% desde compra",
                "urgencia": "MEDIA"
            })
            enviadas["alerta_alta"] = hoy
        
        # Guardar alertas enviadas
        cartera["alertas_enviadas"][ticker] = enviadas
    
    guardar_cartera(cartera)
    return alertas


def enviar_alertas(alertas):
    """Envía las alertas por Telegram."""
    for alerta in alertas:
        mensaje = f"""
{'🚨' if alerta['urgencia'] == 'CRITICA' else '📊'} <b>{alerta['tipo']}</b>

📈 <b>{alerta['ticker']}</b>
💰 Precio actual: ${alerta['precio_actual']:.2f}
📍 Nivel activado: ${alerta['nivel']:.2f}
📊 Ganancia/Pérdida: {alerta['ganancia_pct']:+.2f}%

💡 {alerta['mensaje']}
"""
        enviar_telegram(mensaje)
        print(f"📤 Alerta enviada: {alerta['ticker']} - {alerta['tipo']}")


def vigilar(intervalo_minutos=5):
    """Modo vigilancia continua."""
    config = cargar_config()
    intervalo = config.get("intervalo_minutos", intervalo_minutos)
    
    print(f"\n👁️ Modo vigilancia activado")
    print(f"   Comprobando cada {intervalo} minutos")
    print(f"   Pulsa Ctrl+C para detener\n")
    
    # Enviar mensaje de inicio
    enviar_telegram("🟢 <b>Vigilancia activada</b>\n\nRecibirás alertas cuando tus acciones alcancen los niveles configurados.")
    
    while True:
        try:
            print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Comprobando precios...")
            
            alertas = comprobar_alertas()
            
            if alertas:
                enviar_alertas(alertas)
            else:
                print("   ✅ Sin alertas")
            
            print(f"   Próxima comprobación en {intervalo} minutos...")
            time.sleep(intervalo * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Vigilancia detenida")
            enviar_telegram("🔴 <b>Vigilancia detenida</b>")
            break


# --------------------------------------------------
# INTERFAZ DE LÍNEA DE COMANDOS
# --------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema de alertas de trading con Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python alertas_telegram.py --setup                    Configurar Telegram
  python alertas_telegram.py --comprar AAPL 150.50 10   Registrar compra
  python alertas_telegram.py --vender AAPL              Eliminar posición
  python alertas_telegram.py --cartera                  Ver cartera
  python alertas_telegram.py --vigilar                  Iniciar vigilancia
  python alertas_telegram.py --test                     Enviar mensaje de prueba
        """
    )
    
    parser.add_argument("--setup", action="store_true", help="Configurar Telegram")
    parser.add_argument("--comprar", nargs=3, metavar=("TICKER", "PRECIO", "CANTIDAD"),
                       help="Registrar una compra")
    parser.add_argument("--vender", metavar="TICKER", help="Eliminar posición")
    parser.add_argument("--cartera", action="store_true", help="Ver cartera actual")
    parser.add_argument("--vigilar", action="store_true", help="Iniciar vigilancia")
    parser.add_argument("--comprobar", action="store_true", help="Comprobar alertas una vez")
    parser.add_argument("--test", action="store_true", help="Enviar mensaje de prueba")
    parser.add_argument("--intervalo", type=int, default=5, help="Intervalo en minutos")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_telegram()
        
    elif args.comprar:
        ticker, precio, cantidad = args.comprar
        añadir_compra(ticker, float(precio), int(cantidad))
        
    elif args.vender:
        eliminar_posicion(args.vender)
        
    elif args.cartera:
        ver_cartera()
        
    elif args.vigilar:
        vigilar(args.intervalo)
        
    elif args.comprobar:
        alertas = comprobar_alertas()
        if alertas:
            enviar_alertas(alertas)
        else:
            print("✅ Sin alertas activas")
            
    elif args.test:
        if enviar_telegram("🧪 <b>Mensaje de prueba</b>\n\n¡Tu bot funciona correctamente!"):
            print("✅ Mensaje enviado, revisa Telegram")
        else:
            print("❌ Error. Ejecuta --setup para configurar")
            
    else:
        # Sin argumentos: mostrar cartera y comprobar
        ver_cartera()
        print("\n" + "-"*50)
        print("Comprobando alertas...")
        alertas = comprobar_alertas()
        if alertas:
            for a in alertas:
                print(f"  {a['tipo']} {a['ticker']}: ${a['precio_actual']:.2f}")


if __name__ == "__main__":
    main()
