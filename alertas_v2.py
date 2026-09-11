"""
alertas_v2.py — Alertas de Telegram para las señales de compra/venta del motor v2.

QUÉ AVISA
---------
Cada valor vigilado está siempre en un estado: COMPRA (posición larga viva),
VENTA (posición corta viva) o MANTENER (fuera de mercado). El estado lo
determina la estrategia de bandas de Bollinger y RSI con todas sus reglas:
entrada por las cuatro condiciones, y salida por stop, por objetivo en la
banda media o por salida temporal.

El aviso se manda SOLO al entrar en compra o en venta:

    MANTENER -> COMPRA     🟢 aviso de compra
    MANTENER -> VENTA      🔴 aviso de venta

Salir de una posición (COMPRA o VENTA -> MANTENER) NO genera aviso, aunque el
estado cambie. El estado de cada valor se guarda entre ejecuciones en un JSON,
así que si la situación no cambia no llega nada.

NOTA SOBRE TELEGRAM
-------------------
La API de bots NO permite enviar por número de teléfono ni por @alias: es una
protección antispam. Un bot solo puede escribir a quien le haya escrito antes,
y lo identifica por un chat_id numérico. Por eso aquí se guarda el chat_id y
se le pone un alias legible para manejarlo en la interfaz. La función
`obtener_chats_recientes()` lo saca automáticamente de los /start recibidos.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from signals import CONFIG_V2, generar_senales, normalizar_ohlc

DIR_DATOS = Path(__file__).resolve().parent / "data"
FICHERO_CONFIG = DIR_DATOS / "alertas_config.json"
FICHERO_ESTADO = DIR_DATOS / "alertas_estado.json"

API = "https://api.telegram.org/bot{token}/{metodo}"
TIMEOUT = 20


# ======================================================================
# TOKEN
# ======================================================================

def obtener_token() -> str | None:
    """
    Busca el token del bot en este orden:
      1. Variable de entorno TELEGRAM_BOT_TOKEN  (GitHub Actions)
      2. st.secrets["TELEGRAM_BOT_TOKEN"]        (Streamlit Cloud)
      3. El que ya tengas en alertas_telegram.py (reutiliza tu configuración)
    """
    t = os.environ.get("TELEGRAM_BOT_TOKEN")
    if t:
        return t.strip()

    try:
        import streamlit as st
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            return str(st.secrets["TELEGRAM_BOT_TOKEN"]).strip()
    except Exception:
        pass

    try:
        import alertas_telegram as at
        for nombre in ("TELEGRAM_TOKEN", "BOT_TOKEN", "TOKEN", "TELEGRAM_BOT_TOKEN"):
            v = getattr(at, nombre, None)
            if isinstance(v, str) and ":" in v:
                return v.strip()
    except Exception:
        pass

    return None


# ======================================================================
# CONFIGURACIÓN Y ESTADO
# ======================================================================

CONFIG_VACIA = {
    "destinatarios": [],   # [{"alias": "Pedro", "chat_id": "123456789"}]
    "valores": [],         # [{"ticker": "SAN.MC", "compra": True, "venta": True}]
    # Los cortos vienen activados para las alertas: sin esto nunca existiría
    # el estado VENTA y solo llegarían avisos de compra.
    "parametros": {"permitir_short": True},
    "periodo": "2y",
}


def cargar_config(path: Path | str = FICHERO_CONFIG) -> dict:
    p = Path(path)
    if not p.exists():
        return dict(CONFIG_VACIA)
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return {**CONFIG_VACIA, **cfg}
    except Exception:
        return dict(CONFIG_VACIA)


def guardar_config(cfg: dict, path: Path | str = FICHERO_CONFIG) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


def cargar_estado(path: Path | str = FICHERO_ESTADO) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def guardar_estado(estado: dict, path: Path | str = FICHERO_ESTADO) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


# ======================================================================
# TELEGRAM
# ======================================================================

def enviar_mensaje(chat_id: str, texto: str, token: str | None = None) -> tuple[bool, str]:
    token = token or obtener_token()
    if not token:
        return False, "No hay token del bot configurado."
    try:
        r = requests.post(
            API.format(token=token, metodo="sendMessage"),
            json={"chat_id": str(chat_id), "text": texto,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=TIMEOUT,
        )
        d = r.json()
        if d.get("ok"):
            return True, "Enviado"
        return False, str(d.get("description", "Error desconocido"))
    except Exception as e:
        return False, str(e)


def obtener_chats_recientes(token: str | None = None) -> list[dict]:
    """
    Devuelve los chats que han escrito al bot recientemente (los /start).
    Es la forma de conseguir el chat_id sin pedírselo a nadie: la persona
    escribe al bot y aquí aparece su alias y su id.
    """
    token = token or obtener_token()
    if not token:
        return []
    try:
        r = requests.get(API.format(token=token, metodo="getUpdates"), timeout=TIMEOUT)
        d = r.json()
        if not d.get("ok"):
            return []
        vistos, salida = set(), []
        for upd in d.get("result", []):
            msg = upd.get("message") or upd.get("edited_message") or {}
            chat = msg.get("chat") or {}
            cid = chat.get("id")
            if cid is None or cid in vistos:
                continue
            vistos.add(cid)
            nombre = (chat.get("title")
                      or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
                      or chat.get("username") or str(cid))
            salida.append({
                "chat_id": str(cid),
                "alias": nombre,
                "username": chat.get("username", ""),
                "tipo": chat.get("type", ""),
            })
        return salida
    except Exception:
        return []


def comprobar_bot(token: str | None = None) -> tuple[bool, str]:
    token = token or obtener_token()
    if not token:
        return False, "No hay token configurado."
    try:
        r = requests.get(API.format(token=token, metodo="getMe"), timeout=TIMEOUT)
        d = r.json()
        if d.get("ok"):
            u = d["result"].get("username", "?")
            return True, f"@{u}"
        return False, str(d.get("description", "Token no válido"))
    except Exception as e:
        return False, str(e)


# ======================================================================
# ESTADO DE LA ESTRATEGIA (máquina de estados)
# ======================================================================
#
# Cada valor está SIEMPRE en uno de tres estados:
#
#   COMPRA    hay una posición larga viva según la estrategia
#   VENTA     hay una posición corta viva (solo si permites cortos)
#   MANTENER  fuera de mercado
#
# Los avisos se disparan en la TRANSICIÓN, no cada día:
#
#   MANTENER -> COMPRA    evento COMPRA
#   MANTENER -> VENTA      evento VENTA
#
# Salir de una posición (COMPRA o VENTA -> MANTENER) cambia el estado pero NO
# genera aviso. Solo se comunican las entradas.
# ======================================================================

def descargar(ticker: str, periodo: str = "2y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        d = yf.Ticker(ticker).history(period=periodo, auto_adjust=True)
        if d is None or d.empty:
            return None
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return normalizar_ohlc(d)
    except Exception:
        return None


def recorrer_estados(sen: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Recorre el histórico aplicando las reglas completas de la estrategia
    (entrada, stop, objetivo en la banda media y salida temporal) y devuelve
    el estado barra a barra. Es la misma mecánica que el backtest, pero sin
    contabilidad: aquí solo interesa en qué estado está el valor.
    """
    idx = sen.index
    o = sen["Open"].to_numpy(float)
    h = sen["High"].to_numpy(float)
    l = sen["Low"].to_numpy(float)
    c = sen["Close"].to_numpy(float)
    mid = sen["bb_mid"].to_numpy(float)
    senal = sen["senal"].to_numpy(object)
    stop_ref = sen["stop_ref"].to_numpy(float)

    historia = []
    pos = None

    for i in range(len(sen)):
        evento, motivo = None, ""

        if pos is not None:
            pos["barras"] += 1
            largo = pos["lado"] == "LONG"
            objetivo = mid[i] if cfg.get("objetivo_dinamico", True) else pos["tp1"]

            tocado_stop = (l[i] <= pos["stop"]) if largo else (h[i] >= pos["stop"])
            tocado_obj = ((h[i] >= objetivo) if largo else (l[i] <= objetivo)) \
                if pd.notna(objetivo) else False

            # Las SALIDAS cambian el estado a MANTENER pero NO generan aviso.
            # Solo se comunica la entrada en compra o en venta.
            if tocado_stop:
                motivo = "cierre: stop de protección"
                pos = None
            elif tocado_obj:
                motivo = "cierre: objetivo alcanzado (banda media)"
                pos = None
            elif pos["barras"] >= cfg.get("max_barras", 10):
                motivo = "cierre: salida temporal, no ha revertido"
                pos = None

        if pos is None and evento is None and senal[i] in ("LONG", "SHORT"):
            largo = senal[i] == "LONG"
            pos = {"lado": senal[i], "stop": stop_ref[i], "tp1": mid[i],
                   "barras": -1, "entrada": c[i], "fecha": idx[i]}
            evento = "COMPRA" if largo else "VENTA"
            motivo = "se cumplen las condiciones de entrada"

        if pos is None:
            estado = "MANTENER"
        else:
            estado = "COMPRA" if pos["lado"] == "LONG" else "VENTA"

        historia.append({"fecha": idx[i], "estado": estado,
                         "evento": evento, "motivo": motivo})

    return historia


def evaluar_valor(ticker: str, cfg_senales: dict | None = None,
                  periodo: str = "2y", df: pd.DataFrame | None = None) -> dict | None:
    """Estado actual de un valor según la estrategia. None si no hay datos."""
    cfg = {**CONFIG_V2, **(cfg_senales or {})}
    d = df if df is not None else descargar(ticker, periodo)
    if d is None or len(d) < 260:
        return None
    try:
        sen = generar_senales(d, cfg)
    except Exception:
        return None

    historia = recorrer_estados(sen, cfg)
    hoy = historia[-1]
    u = sen.iloc[-1]

    # ¿desde cuándo lleva en este estado?
    dias = 1
    for h in reversed(historia[:-1]):
        if h["estado"] != hoy["estado"]:
            break
        dias += 1

    # niveles de la posición viva, si la hay
    stop = objetivo = float("nan")
    if hoy["estado"] in ("COMPRA", "VENTA"):
        stop = float(u["stop_ref"])
        objetivo = float(u["bb_mid"])

    return {
        "ticker": ticker,
        "fecha": sen.index[-1].strftime("%Y-%m-%d"),
        "estado": hoy["estado"],
        "evento": hoy["evento"],
        "motivo": hoy["motivo"],
        "dias_en_estado": dias,
        "precio": float(u["Close"]),
        "rsi": float(u["rsi"]),
        "bb_low": float(u["bb_low"]),
        "bb_mid": float(u["bb_mid"]),
        "bb_up": float(u["bb_up"]),
        "pct_b": float(u["bb_pct_b"]) * 100 if pd.notna(u["bb_pct_b"]) else float("nan"),
        "tendencia": str(u["tendencia"]),
        "stop": stop,
        "objetivo": objetivo,
        "rr": float(u["rr1"]) if pd.notna(u["rr1"]) else float("nan"),
    }


def evento_por_transicion(estado_previo: str, estado_actual: str) -> str | None:
    """
    Deduce qué operación implica un cambio de estado. Sirve para no perder
    avisos si el revisor no se ejecutó algún día (fallo de red, mercado
    cerrado, la acción de GitHub caída): al volver, compara el estado
    guardado con el de hoy y comunica lo que haya pasado en medio.
    """
    if estado_previo == estado_actual:
        return None
    if estado_actual == "COMPRA":
        return "COMPRA"
    if estado_actual == "VENTA":
        return "VENTA"
    # Pasar a MANTENER (cerrar una posición) no genera aviso.
    return None


def hay_cambio(estado_previo: dict | None, actual: dict) -> bool:
    """
    Avisa solo si HOY se ha producido un evento de compra o de venta y el
    estado ha cambiado respecto a lo último registrado. Así el aviso llega
    una vez, no cada día que la situación siga igual, y tampoco se repite si
    el revisor se ejecuta dos veces la misma jornada.
    """
    if actual.get("evento") not in ("COMPRA", "VENTA"):
        return False
    if estado_previo is None:
        return True
    return estado_previo.get("estado") != actual["estado"]


# ======================================================================
# MENSAJE
# ======================================================================

def construir_mensaje(a: dict) -> str:
    compra = a["evento"] == "COMPRA"
    icono = "🟢" if compra else "🔴"
    titulo = "COMPRAR" if compra else "VENDER"

    lineas = [
        f"{icono} <b>{titulo}</b> · <b>{a['ticker']}</b>",
        f"<i>{a['fecha']}</i>",
        "",
        f"Motivo: {a['motivo']}",
        f"Estado: pasa a <b>{a['estado']}</b>",
        "",
        f"Precio: <b>{a['precio']:,.2f}</b>",
        f"Bandas: {a['bb_low']:,.2f} · {a['bb_mid']:,.2f} · {a['bb_up']:,.2f}  (%B {a['pct_b']:.0f})",
        f"RSI: {a['rsi']:.1f}   Tendencia: {a['tendencia']}",
    ]

    if a["estado"] in ("COMPRA", "VENTA") and pd.notna(a["stop"]):
        riesgo = abs(a["precio"] - a["stop"])
        lineas += [
            "",
            f"Stop: <b>{a['stop']:,.2f}</b>  (riesgo {riesgo:,.2f} por título)",
            f"Objetivo: <b>{a['objetivo']:,.2f}</b>  (banda media)",
            f"R/R: {a['rr']:.2f}",
        ]

    lineas += ["", "<i>La operación se ejecuta en la apertura de la sesión siguiente.</i>"]
    return "\n".join(lineas)


# ======================================================================
# EJECUCIÓN COMPLETA
# ======================================================================

def revisar(config: dict | None = None, enviar: bool = True,
            path_estado: Path | str = FICHERO_ESTADO,
            token: str | None = None,
            funcion_envio=None) -> dict:
    """
    Revisa todos los valores de la lista, detecta cambios y manda los avisos.

    `funcion_envio` permite inyectar un sustituto de `enviar_mensaje` en pruebas.
    Devuelve un resumen con lo evaluado, lo avisado y los errores.
    """
    cfg = config or cargar_config()
    estado = cargar_estado(path_estado)
    envio = funcion_envio or enviar_mensaje
    cfg_senales = {**CONFIG_V2, **(cfg.get("parametros") or {})}
    periodo = cfg.get("periodo", "2y")

    resumen = {"evaluados": [], "alertas": [], "errores": [],
               "envios": [], "momento": datetime.now().isoformat(timespec="seconds")}

    for v in cfg.get("valores", []):
        tk = v.get("ticker", "").strip().upper()
        if not tk:
            continue
        actual = evaluar_valor(tk, cfg_senales, periodo)
        if actual is None:
            resumen["errores"].append(f"{tk}: sin datos suficientes")
            continue
        resumen["evaluados"].append(actual)

        # Si el estado cambió mientras el revisor no se ejecutaba, se recupera
        previo = estado.get(tk)
        if actual["evento"] is None and previo:
            inferido = evento_por_transicion(previo.get("estado", ""), actual["estado"])
            if inferido:
                actual = {**actual, "evento": inferido,
                          "motivo": "cambio detectado desde la última revisión"}

        quiere = ((actual["evento"] == "COMPRA" and v.get("compra", True))
                  or (actual["evento"] == "VENTA" and v.get("venta", True)))

        if quiere and hay_cambio(estado.get(tk), actual):
            resumen["alertas"].append(actual)
            if enviar:
                texto = construir_mensaje(actual)
                for d in cfg.get("destinatarios", []):
                    ok, det = envio(d["chat_id"], texto, token)
                    resumen["envios"].append(
                        {"alias": d.get("alias", d["chat_id"]), "ticker": tk,
                         "ok": ok, "detalle": det})

        estado[tk] = {"estado": actual["estado"], "fecha": actual["fecha"],
                      "precio": actual["precio"],
                      "dias_en_estado": actual["dias_en_estado"]}

    guardar_estado(estado, path_estado)
    return resumen
