from openpilot.selfdrive.carrot import cluster_autorun


def test_device_cluster_uses_standalone_carrot_navi_ipc():
  args = cluster_autorun._cluster_args(
    hud_mode=1,
    configured_encoder_mode=cluster_autorun.ENCODER_AUTO,
    active_encoder_mode=cluster_autorun.ENCODER_JPEG,
    core_mode=cluster_autorun.CORE_MODE_DEDICATED,
    priority=10,
  )

  assert args[:2] == ["--input", "live"]
  assert "--navi-overlay" not in args
  assert "--navi-publish-cereal" not in args
  assert args[args.index("--output") + 1] == "usb"
  assert float(args[args.index("--duration") + 1]) == cluster_autorun.JPEG_FALLBACK_H264_RETRY_S


def test_explicit_jpeg_does_not_retry_h264():
  args = cluster_autorun._cluster_args(
    hud_mode=1,
    configured_encoder_mode=cluster_autorun.ENCODER_JPEG,
    active_encoder_mode=cluster_autorun.ENCODER_JPEG,
    core_mode=cluster_autorun.CORE_MODE_DEDICATED,
    priority=10,
  )

  assert "--duration" not in args
