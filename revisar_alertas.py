#!/usr/bin/env python3
"""
revisar_alertas.py — Ejecutor de las alertas. Pensado para GitHub Actions o cron.

Uso:
    python revisar_alertas.py                # revisa y envía
    python revisar_alertas.py --simular      # revisa sin enviar nada
    python revisar_alertas.py --config otro.json

El token del bot se lee de la variable de entorno TELEGRAM_BOT_TOKEN.

Por qué existe: una app de Streamlit Cloud se duerme cuando nadie la visita, así
que las alertas no pueden depender de que la app esté abierta. Este script se
ejecuta solo al cierre de mercado y actualiza el fichero de estado.
"""

import argparse
import sys
from pathlib import Path

import alertas_v2 as A


def main():
    p = argparse.ArgumentParser(description="Revisa señales y envía alertas de Telegram")
    p.add_argument("--config", default=str(A.FICHERO_CONFIG))
    p.add_argument("--estado", default=str(A.FICHERO_ESTADO))
    p.add_argument("--simular", action="store_true", help="No envía nada")
    args = p.parse_args()

    cfg = A.cargar_config(args.config)

    if not cfg.get("valores"):
        print("No hay valores configurados. Nada que revisar.")
        return 0
    if not cfg.get("destinatarios") and not args.simular:
        print("No hay destinatarios configurados.", file=sys.stderr)
        return 1

    if not args.simular:
        ok, det = A.comprobar_bot()
        if not ok:
            print(f"Bot no disponible: {det}", file=sys.stderr)
            return 1
        print(f"Bot conectado: {det}")

    print(f"Revisando {len(cfg['valores'])} valores…")
    r = A.revisar(cfg, enviar=not args.simular, path_estado=args.estado)

    for e in r["evaluados"]:
        print(f"  {e['ticker']:<12} {e['estado']:<9} "
              f"precio {e['precio']:>10,.2f}  RSI {e['rsi']:>5.1f}  "
              f"%B {e['pct_b']:>6.1f}  ({e['dias_en_estado']} sesiones)")

    for err in r["errores"]:
        print(f"  ! {err}", file=sys.stderr)

    if r["alertas"]:
        print(f"\n{len(r['alertas'])} cambio(s) de estado:")
        for a in r["alertas"]:
            print(f"  -> {a['ticker']}: {a['evento']} a {a['precio']:,.2f} "
                  f"({a['motivo']})")
    else:
        print("\nSin cambios de estado.")

    fallos = [e for e in r["envios"] if not e["ok"]]
    if r["envios"]:
        print(f"\nEnvíos: {len(r['envios']) - len(fallos)} correctos, {len(fallos)} fallidos")
        for f in fallos:
            print(f"  ! {f['alias']} / {f['ticker']}: {f['detalle']}", file=sys.stderr)

    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
