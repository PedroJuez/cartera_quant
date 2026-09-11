"""
alertas_ui.py — Panel de configuración de las alertas de Telegram.

Aquí se dan de alta los valores a vigilar y las personas que reciben el aviso.
El envío automático lo hace `revisar_alertas.py` desde GitHub Actions; este
panel sirve para configurar, probar y ver el estado.

Autor: Pedro Juez Martel
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import alertas_v2 as A
from signals import CONFIG_V2


def _cfg():
    if "alertas_cfg" not in st.session_state:
        st.session_state["alertas_cfg"] = A.cargar_config()
    return st.session_state["alertas_cfg"]


def render_alertas():
    st.title("🔔 Alertas de compra y venta por Telegram")
    st.caption("Cada valor está en COMPRA, VENTA o MANTENER según la estrategia de "
               "bandas y RSI. El aviso llega en el momento del cambio, indicando la "
               "operación a realizar. Pasar a MANTENER no genera mensaje por sí solo.")

    cfg = _cfg()

    # ------------------------------------------------ estado del bot
    token = A.obtener_token()
    if not token:
        st.error(
            "**No hay token del bot configurado.** Añádelo en uno de estos sitios:\n\n"
            "- En Streamlit Cloud: *Settings → Secrets* → `TELEGRAM_BOT_TOKEN = \"123:ABC…\"`\n"
            "- En GitHub: *Settings → Secrets and variables → Actions* → `TELEGRAM_BOT_TOKEN`\n"
            "- O se reutiliza automáticamente el que ya tengas en `alertas_telegram.py`"
        )
    else:
        ok, det = A.comprobar_bot(token)
        if ok:
            st.success(f"Bot conectado: **{det}**")
        else:
            st.error(f"El token no funciona: {det}")

    t1, t2, t3, t4 = st.tabs(["📋 Valores", "👥 Destinatarios", "🧪 Probar", "⚙️ Publicar"])

    # ============================================================ VALORES
    with t1:
        st.markdown("#### Valores vigilados")

        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        nuevos = c1.text_input("Tickers (separados por coma)", placeholder="SAN.MC, ITX.MC, SPY")
        av_compra = c2.checkbox("Compra", value=True, key="nv_compra")
        av_venta = c3.checkbox("Venta", value=True, key="nv_venta")
        if c4.button("➕ Añadir", width='stretch'):
            existentes = {v["ticker"] for v in cfg["valores"]}
            añadidos = []
            for t in [x.strip().upper() for x in nuevos.split(",") if x.strip()]:
                if t not in existentes:
                    cfg["valores"].append({"ticker": t, "compra": av_compra, "venta": av_venta})
                    añadidos.append(t)
            if añadidos:
                st.success(f"Añadidos: {', '.join(añadidos)}")
                st.rerun()
            elif nuevos.strip():
                st.info("Ya estaban en la lista.")

        if cfg["valores"]:
            ed = st.data_editor(
                pd.DataFrame(cfg["valores"]),
                width='stretch', hide_index=True, num_rows="dynamic",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", required=True),
                    "compra": st.column_config.CheckboxColumn("Avisar compra"),
                    "venta": st.column_config.CheckboxColumn("Avisar venta"),
                },
                key="editor_valores",
            )
            cfg["valores"] = [
                {"ticker": str(r["ticker"]).strip().upper(),
                 "compra": bool(r.get("compra", True)),
                 "venta": bool(r.get("venta", True))}
                for _, r in ed.iterrows() if str(r.get("ticker", "")).strip()
            ]
            st.caption(f"{len(cfg['valores'])} valor(es). Puedes editar o borrar filas "
                       "directamente en la tabla.")
        else:
            st.info("Todavía no hay valores. Añade alguno arriba.")

        with st.expander("Parámetros de la señal"):
            st.caption("Por defecto se usan los mismos que la pestaña Estrategia v2. "
                       "Cámbialos solo si quieres que las alertas sean más o menos exigentes.")
            p = cfg.get("parametros", {})
            cc = st.columns(3)
            p["rsi_oversold"] = cc[0].slider("RSI sobreventa", 10.0, 45.0,
                                             float(p.get("rsi_oversold", CONFIG_V2["rsi_oversold"])), 1.0)
            p["rsi_overbought"] = cc[1].slider("RSI sobrecompra", 55.0, 90.0,
                                               float(p.get("rsi_overbought", CONFIG_V2["rsi_overbought"])), 1.0)
            p["min_rr"] = cc[2].slider("R/R mínimo", 0.5, 4.0,
                                       float(p.get("min_rr", CONFIG_V2["min_rr"])), 0.1)
            p["usar_filtro_vol"] = st.checkbox(
                "Filtro de régimen de volatilidad",
                bool(p.get("usar_filtro_vol", CONFIG_V2["usar_filtro_vol"])))
            p["permitir_short"] = st.checkbox(
                "Permitir señales de venta en corto",
                bool(p.get("permitir_short", CONFIG_V2["permitir_short"])),
                help="Desactivado, solo avisará de compras.")
            cfg["parametros"] = p

    # ====================================================== DESTINATARIOS
    with t2:
        st.markdown("#### Quién recibe los avisos")
        st.info(
            "**Telegram no permite enviar por teléfono ni por @alias.** Es una "
            "protección antispam de su API: un bot solo puede escribir a quien le "
            "haya escrito antes, y lo identifica por un `chat_id` numérico.\n\n"
            "**Cómo dar de alta a alguien:** que abra tu bot en Telegram y le envíe "
            "`/start`. Después pulsa el botón de abajo y aparecerá aquí para añadirlo "
            "con un clic. No hace falta que nadie busque su número."
        )

        if st.button("🔄 Buscar quién ha escrito al bot", type="primary"):
            chats = A.obtener_chats_recientes(token)
            st.session_state["chats_detectados"] = chats
            if not chats:
                st.warning("Nadie ha escrito al bot recientemente. Que envíe `/start` "
                           "y vuelve a pulsar. Telegram solo guarda los mensajes "
                           "recientes, así que hazlo en el momento.")

        for ch in st.session_state.get("chats_detectados", []):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"**{ch['alias']}**"
                        + (f" · @{ch['username']}" if ch["username"] else ""))
            c2.code(ch["chat_id"], language=None)
            ya = any(d["chat_id"] == ch["chat_id"] for d in cfg["destinatarios"])
            if ya:
                c3.caption("✅ ya está")
            elif c3.button("Añadir", key=f"add_{ch['chat_id']}"):
                cfg["destinatarios"].append({"alias": ch["alias"], "chat_id": ch["chat_id"]})
                st.rerun()

        st.divider()
        st.markdown("**Añadir a mano** (si ya conoces el chat_id)")
        c1, c2, c3 = st.columns([2, 2, 1])
        alias_m = c1.text_input("Alias", placeholder="Pedro")
        cid_m = c2.text_input("chat_id", placeholder="123456789")
        if c3.button("➕", key="add_manual") and cid_m.strip():
            cfg["destinatarios"].append({"alias": alias_m.strip() or cid_m.strip(),
                                         "chat_id": cid_m.strip()})
            st.rerun()

        st.divider()
        if cfg["destinatarios"]:
            st.markdown("**Destinatarios actuales**")
            for i, d in enumerate(list(cfg["destinatarios"])):
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(f"**{d['alias']}**")
                c2.code(d["chat_id"], language=None)
                if c3.button("🗑️", key=f"del_{i}"):
                    cfg["destinatarios"].pop(i)
                    st.rerun()
        else:
            st.warning("Sin destinatarios: no se enviará nada.")

    # ============================================================= PROBAR
    with t3:
        st.markdown("#### Probar")

        c1, c2 = st.columns(2)
        if c1.button("📨 Enviar mensaje de prueba", width='stretch'):
            if not cfg["destinatarios"]:
                st.error("No hay destinatarios.")
            else:
                for d in cfg["destinatarios"]:
                    ok, det = A.enviar_mensaje(
                        d["chat_id"],
                        "✅ <b>Prueba de alertas</b>\n\nSi lees esto, la configuración "
                        "es correcta y recibirás los avisos de compra y venta.",
                        token)
                    (st.success if ok else st.error)(f"{d['alias']}: {det}")

        if c2.button("🔍 Revisar ahora sin enviar", width='stretch'):
            if not cfg["valores"]:
                st.error("No hay valores configurados.")
            else:
                with st.spinner("Evaluando…"):
                    r = A.revisar(cfg, enviar=False, funcion_envio=lambda *a, **k: (True, ""))
                if r["evaluados"]:
                    df = pd.DataFrame(r["evaluados"])[
                        ["ticker", "fecha", "estado", "evento", "dias_en_estado",
                         "precio", "rsi", "pct_b", "tendencia", "stop", "objetivo"]]
                    st.dataframe(df.round(2), width='stretch', hide_index=True)
                vivas = [a["ticker"] for a in r["alertas"]]
                if vivas:
                    st.success(f"Señales nuevas: {', '.join(vivas)}")
                    for a in r["alertas"]:
                        st.markdown(A.construir_mensaje(a).replace("<b>", "**")
                                    .replace("</b>", "**").replace("<i>", "_")
                                    .replace("</i>", "_"))
                else:
                    st.info("Sin señales nuevas ahora mismo.")
                for e in r["errores"]:
                    st.warning(e)

        st.divider()
        estado = A.cargar_estado()
        if estado:
            st.markdown("**Último estado registrado**")
            st.dataframe(pd.DataFrame(estado).T.reset_index().rename(
                columns={"index": "ticker"}), width='stretch', hide_index=True)
            st.caption("Solo se avisa cuando el estado cambia respecto a esto. "
                       "Si el revisor no se ejecuta algún día, al volver detecta el "
                       "cambio comparando con este registro y no se pierde el aviso.")

    # =========================================================== PUBLICAR
    with t4:
        st.markdown("#### Publicar la configuración")
        st.warning(
            "**Streamlit Cloud no guarda cambios en el disco.** Lo que edites aquí "
            "se pierde al reiniciarse la app. Para que las alertas automáticas usen "
            "esta configuración, hay que subir el fichero al repositorio."
        )

        texto = json.dumps(cfg, indent=2, ensure_ascii=False)
        st.code(texto, language="json")

        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Descargar alertas_config.json", texto.encode("utf-8"),
                           "alertas_config.json", "application/json", width='stretch')
        if c2.button("💾 Guardar en disco (solo en local)", width='stretch'):
            if A.guardar_config(cfg):
                st.success(f"Guardado en `{A.FICHERO_CONFIG}`")
            else:
                st.error("No se pudo escribir el fichero.")

        st.markdown(
            "**Pasos:** descarga el fichero, colócalo en `data/alertas_config.json` "
            "del repositorio, y haz commit. La acción programada de GitHub lo leerá "
            "en la siguiente ejecución."
        )
