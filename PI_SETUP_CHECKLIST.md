# Raspberry Pi Setup Checklist — Guardian Edge Collector

Do all of this **before the SD card ever goes into the Pi**. Every item here
was learned the hard way (2026-08-13, see git history/session notes around
that date): skipping any one of them turns into an hours-long, blind,
over-the-network debugging session instead of a working Pi on first boot.

See `EDGE_ARCHITECTURE.md` for the *why* (what role this device plays);
this doc is purely the *how* of getting one physically set up.

## Before you flash

Get a monitor + keyboard for the Pi, even temporarily borrowed. It's not
required if the checklist below is followed correctly, but it's the
difference between seeing an error immediately at boot versus diagnosing
it blind over SSH/mDNS/log-file archaeology. Cheap insurance.

## Flashing (Raspberry Pi Imager)

**Use the advanced settings screen (⚙️ gear icon, or Ctrl+Shift+X) before
writing the image.** This is the single highest-leverage step — it moves
every piece of first-boot config into the image itself instead of needing
to be patched in after the fact. Set all of:

- [ ] **Enable SSH**, with password auth *or* (better) paste in a public
      key directly (`~/.ssh/id_ed25519.pub` on the machine you'll SSH
      from). Key-based auth avoids ever needing to know/type a
      first-login password.
- [ ] **Set username + password anyway**, even if using a key. Don't
      assume a distro's "default user" (`pi`, `ubuntu`, etc.) — it varies
      by image and isn't always what generic boilerplate text suggests.
      Setting it explicitly here removes the ambiguity entirely.
- [ ] **Configure WiFi** (SSID + password) if it won't be on Ethernet from
      a router (a direct point-to-point cable to a laptop does **not**
      give it real network access — see "Networking gotcha" below).
- [ ] **Set a distinctive hostname** — don't leave it as `raspberrypi`,
      especially once there's more than one Pi (multi-site future plan in
      `EDGE_ARCHITECTURE.md`). Collisions on the default hostname make
      mDNS discovery ambiguous.
      **If this was skipped and the Pi already booted** (confirmed working
      2026-08-14, renaming `raspberrypi` → `guardian-proto-1` post-hoc):
      `preserve_hostname: false` in `/etc/cloud/cloud.cfg` looks like it'll
      fight you, but this image's cloud-init is pre-baked "already
      completed" (see Recovering section below) so it won't actually
      re-run and revert the change. Just: `sudo hostnamectl set-hostname
      <new-name>`, then manually fix `/etc/hosts` (`hostnamectl` does
      *not* update the `127.0.1.1` line — do it yourself or `sudo` will
      print DNS-resolution warnings), then `sudo systemctl restart
      avahi-daemon` so the new `<new-name>.local` mDNS name is reachable
      immediately instead of only after next boot. SSH host key is
      unchanged (tied to the machine, not the name) — new connections
      just need one `accept-new` for the not-yet-seen hostname string.

If you skip this screen and flash a plain image, everything below still
works, but expect the "recovering from a plain image" path (bottom of this
doc) instead of a clean first boot.

## First boot

- [ ] Power on with the card already configured above. No monitor/keyboard
      needed if the advanced settings were actually applied — SSH should
      be reachable within ~60-90 seconds via `<hostname>.local`.
- [ ] Verify you can SSH in **with the account you actually intended**
      (not an assumed default). `ssh <user>@<hostname>.local`.
- [ ] Check the clock: `date`. The Pi has no RTC — it needs real internet
      access before NTP can correct it. If it's on WiFi with a working
      gateway, this self-corrects within seconds of boot; if it's stuck on
      an old date, that's a sign networking isn't actually working yet.
- [ ] If WiFi doesn't come up: check `nmcli device status` — a `wlan0`
      stuck on `unavailable` usually means the radio is rfkill-blocked
      pending a regulatory country being set. `raspi-config nonint
      do_wifi_country <CC>` is the normal fix, **but verify it actually
      worked** (`cat /sys/class/rfkill/*/soft` — `0` means unblocked) since
      that command silently no-ops if the `iw` package isn't installed on
      the image. If it's missing and there's no internet yet to install it
      (chicken/egg), unblock directly:
      `echo 0 | sudo tee /sys/class/rfkill/<N>/soft` (find the right index
      via `cat /sys/class/rfkill/*/name`).

## Networking gotcha: don't rely on a direct cable for real connectivity

A USB-Ethernet dock cable straight from a laptop to the Pi, with no router
in between, only ever gives you an IPv6 link-local address — enough for
SSH to bootstrap with, but **not** a real network path (no gateway, no
internet, not reachable from anything else on the LAN). Get it onto real
WiFi or a switched Ethernet network as soon as possible; don't try to do
ongoing work over the bootstrap cable.

If you do need the cable temporarily and the host laptop's Ethernet
profile keeps flapping/losing its address: that's usually the laptop's
NetworkManager tearing down the whole interface after a DHCPv4 timeout
(there's no DHCP server on a direct cable). Fix on the laptop side:
`nmcli connection modify "<profile name>" ipv4.method link-local`.

## Recovering from a plain (unconfigured) image

If a card was already flashed without the advanced settings above, or was
handed to you that way:

1. Check what first-boot mechanism the image actually uses before assuming
   anything — mount the boot partition (`/boot` or `/boot/firmware`) and
   look. A bare `ssh` file present/absent means the classic Raspberry Pi OS
   mechanism; `user-data`/`network-config`/`meta-data` files mean
   `cloud-init` (NoCloud datasource) instead — different fix for each.
2. **cloud-init images specifically:** if the Pi has already booted once
   before you edit `user-data`/`network-config`, your edits will be
   silently ignored on the next boot unless you also clear cloud-init's
   "already done" state first: on the root partition,
   `sudo rm -rf /var/lib/cloud/{instances,data,sem,instance}`. Do this
   every time you edit either file after a first boot has happened.
3. **Don't trust boilerplate comments in a stock `user-data` template** —
   they may describe a generic default (e.g. "creates user `ubuntu`") that
   doesn't match this specific image's actual default user. Check
   `/etc/cloud/cloud.cfg`'s `system_info.default_user.name` on the root
   partition to know the real default before assuming.
4. **To add an SSH key to a specific user via `user-data`, be explicit** —
   a top-level `ssh_authorized_keys:` key doesn't reliably land on the
   distro's actual default user on every image (it landed on `root` on a
   `raspberry-pi-os`-flavored `cloud.cfg`). Use an explicit list instead:
   ```yaml
   users:
   - default
   - name: <username>
     ssh_authorized_keys:
     - ssh-ed25519 AAAA... your-key-comment
   ```
5. Batch all needed edits (SSH, user/key, WiFi) into one card session
   before putting it back in the Pi — discovering requirements one at a
   time means a physical card swap for each one.
