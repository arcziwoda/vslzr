# Nasłuchiwanie audio z Samsung HW-Q990F (Spotify Connect / AirPlay 2) bez mikrofonu

*Research date: 2026-05-05. Use case: backend FFT + beat detection (projekt VSLZR). PC nie jest podłączony fizycznie do soundbara — tylko sieć LAN.*

## TL;DR

- **Najprostsza ścieżka na macOS: natywny AirPlay 2 receiver wbudowany w macOS Monterey+ jako sink, BlackHole jako virtual audio cable, VSLZR czyta BlackHole jako audio device.** iPhone AirPlay → grupa {Q990F + MacBook}. Zero shairport-sync, zero NQPTP, zero Pi. Apple sam załatwił PTP i grupowanie. *(Ta opcja umknęła pierwotnemu researchowi — workerzy zostali zaframowani na third-party receiverach typu shairport-sync; native macOS AirPlay sink znajdziesz w System Settings → General → AirDrop & Handoff → AirPlay Receiver.)*
- **Na Windowsie ścieżka "wbudować w aplikację" jest praktycznie zamknięta.** shairport-sync Cygwin = AirPlay 1 only (deal-breaker), goplay2 nie ma żadnych binarek, WSL2 + AirPlay 2 = advanced manual setup, nie embed.
- **Linux/Pi (gdyby ktoś nie miał Maca):** Raspberry Pi 4 z `shairport-sync` v5+ + `nqptp`, PCM do FIFO, VSLZR/skrypt czyta strumień. [Source](https://github.com/mikebrady/shairport-sync/blob/master/README.md)
- **Spotify Connect: niewykonalne równolegle.** Protokół single-active-device — drugi `librespot` na tym samym koncie *kradnie* playback z soundbara, nie mirroruje. [Source](https://github.com/librespot-org/librespot/issues/793) Trzeba więc puszczać z aplikacji Spotify przez **AirPlay** (nie Spotify Connect), żeby grupa {Q990F + Mac} dostała ten sam strumień.
- **`shairport-sync` AirPlay 2 NIE działa bezpośrednio na macOS** (NQPTP vs macOS PTP na UDP 319/320), ale to **i tak nie potrzebne** — macOS ma natywny receiver. [Source](https://github.com/mikebrady/shairport-sync/blob/master/AIRPLAY2.md)
- **Sam Q990F nie wystawia audio przez sieć.** UPnP DMR / SmartThings / Roon Ready — wszystko receive-only. Porty WAM (55001/56001) zamknięte na Q990B+ (i przez ekstrapolację Q990F); port 8080 to tylko control plane (volume/input). [Source](https://community.smartthings.com/t/why-port-56001-is-not-accessible-on-q990b-soundbar/273872)
- **Pasywny sniffing odpada.** AirPlay 2 RTP zaszyfrowane ChaCha20-Poly1305 (klucze z parowania); Spotify Connect leci TLS 1.2+ z chmury bezpośrednio do soundbara (nie przez telefon). [Source](https://emanuelecozzi.net/docs/airplay2/rtp)
- **Spotify Web API jako fallback metadata jest praktycznie martwy** dla nowych aplikacji od 2024-11-27 — `audio-analysis` / `audio-features` (beaty, sekcje) wymagają teraz extended access (≥250k MAU). Zostaje tylko `currently-playing` z laggiem ~500-1000 ms — bezużyteczne dla beat-sync. [Source](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)

## Werdykt według ścieżki

| Ścieżka | Czy da się dostać audio do PC? | Sprzęt dodatkowy | Latencja | Komentarz |
|---|---|---|---|---|
| AirPlay 2 (iPhone/Mac → grupa {Q990F + Pi}) | **TAK** | Pi 4 lub Linux box | ~kilkaset ms (buffering shairport-sync, niesync-krytyczny dla pipe) | Rekomendowana ścieżka |
| Spotify Connect → Q990F + parallel listen | **NIE** | n/a | n/a | Protokół wyklucza |
| Spotify → przerzucić receiver na PC (`librespot --backend pipe`) | TAK, ale soundbar milczy | brak | low | Tracisz soundbar, więc nie spełnia "puszczam przez Spotify Connect" |
| Sniffing pakietów AirPlay/Spotify w LAN | **NIE** | n/a | n/a | E2E encryption |
| Q990F lokalne API (port 8080 / WAM / SmartThings) | **NIE** | n/a | n/a | Tylko control, brak audio |
| Spotify Web API (`/audio-analysis` + `/currently-playing`) | Częściowo (metadata-only) | brak | 500-1000+ ms | Audio-analysis nieosiągalne dla nowych appek; nie nadaje się do real-time |

**Confidence: high** — werdykt opiera się na oficjalnej dokumentacji Spotify (Connect Technical Requirements, Web API blog Nov 2024) i kodzie/dokumentacji shairport-sync.

## Praktyczna rekomendacja #1 (macOS): natywny AirPlay receiver + BlackHole

**Zalecana ścieżka dla użytkownika z MacBookiem.** Apple od macOS Monterey (12, 2021) ma wbudowany AirPlay 2 receiver — Mac sam jest pełnoprawnym członkiem AirPlay 2 multi-room ekosystemu (PTP sync, grupowanie z HomePodami i certyfikowanymi soundbarami jak Q990F). Nie potrzeba żadnego open-source receivera, NQPTP ani Pi.

### Setup

1. **Włącz natywny AirPlay receiver na Macu**: System Settings → General → AirDrop & Handoff → AirPlay Receiver: On (Allow AirPlay for: "Anyone on the same network")
2. **Zainstaluj BlackHole 2ch** (free, open-source virtual audio cable): `brew install blackhole-2ch`
3. **Skieruj output Maca na BlackHole**: System Settings → Sound → Output → BlackHole 2ch (lub utwórz Multi-Output Device w Audio MIDI Setup łączące BlackHole + speakers, jeśli chcesz słyszeć też na Macu)
4. **VSLZR**: wybierz BlackHole 2ch jako audio device (existing AudioCapture device selection)
5. **Puszczasz muzykę z iPhone'a**: Spotify → ikonka AirPlay (nie Spotify Connect) → zaznacz oba: Q990F i MacBook

### Latencja — ważna obserwacja

Pomimo że AirPlay ma ~200-500 ms buffera od "klik play" do "dźwięk wychodzi", soundbar Q990F i MacBook dostają audio **synchronicznie** przez PTP (~1 ms accuracy). Stąd realny dystans między "uszy słyszą beat z soundbara" a "lampa flashuje" = sama latencja pipeline'u VSLZR (~80-120 ms) — **identyczna** co przy mikrofonie, bez akustycznej propagacji w środku. PLL predyktor już to kompensuje.

### Caveaty

- Cross-vendor PTP jitter (Apple ↔ Samsung): zwykle <10 ms, dla beat detection nieistotne
- MacBook nie może iść do sleep'u podczas używania — Power Settings → Wake for network access, lub `caffeinate`
- Pierwszy track po włączeniu: większy buffer, stabilizuje się po 10-15 s

**Confidence: high** dla samej ścieżki (Apple official feature od 4 lat); **medium** dla "BlackHole capture nie ma drift'u względem Q990F" — wymaga empirycznego testu A/B (klaśnięcie + obserwacja flasha).

---

## Praktyczna rekomendacja #2 (Linux/Pi): shairport-sync + nqptp

Sensowne tylko jeśli (a) nie używasz Maca, (b) masz Linux box w sieci, (c) chcesz separować audio gateway od głównego systemu.

### Sprzęt
- Raspberry Pi 4 (4 GB+) na tym samym LAN co Q990F — najtańsza opcja realnego hosta
- Alternatywa: cokolwiek z Linuksem (NUC, mini PC, Linux VM **z `--net=host` na Linux hoście**, nie na Macu)

### Software
- `shairport-sync` v5+ z `--with-airplay-2` i `--with-pipe`
- `nqptp` jako companion daemon (wymagany dla AirPlay 2)
- Bind UDP 319/320 — zarezerwowane dla NQPTP; dlatego nie zadziała na macOS
- Wyjście: pipe `/tmp/shairport-sync-audio` — surowe interleaved PCM 16-bit LE 44.1 kHz / 48 kHz, chunki 352 frames. Jeśli nikt nie czyta, frame'y są droppowane (nieblokujące). [Source](https://github.com/mikebrady/shairport-sync/blob/master/README.md)

### Workflow
1. Pi widoczny na LAN jako AirPlay 2 receiver (np. nazwa "Hue Tap")
2. iPhone/Mac → AirPlay → wybierasz grupę: Q990F + Hue Tap → Q990F gra normalnie, Pi dostaje synchronizowany strumień (PTP, ~1 ms accuracy)
3. Na Pi: proces FFT/beat detection czyta `/tmp/shairport-sync-audio` jako binary file w pętli
4. Pi → WebSocket / HTTP → backend VSLZR (lub uruchom cały VSLZR na Pi)

### Format strumienia
- Z **Apple Music**: AAC 256 kbps przed wysłaniem (Apple downsampluje nawet lossless tracki przy AirPlay 2 do third-party), shairport dekoduje do PCM. [Source](https://darko.audio/2023/10/apple-airplay-isnt-always-lossless-sometimes-its-lossy/)
- Z **lokalnych ALAC**: ALAC → PCM
- Z **Spotify** (po stronie iPhone'a) → AAC/Ogg → PCM. Dla beat detection kompresja AAC 256 jest wystarczająca.

**Confidence: high** dla samego shairport-sync; **medium** dla "grouping {Q990F + shairport} działa out of the box" — zachowanie iOS przy mieszaniu certyfikowanych (Q990F) i niecertyfikowanych (shairport) AirPlay 2 receiverów dawało historycznie różne efekty; warto przetestować empirycznie zanim się zatwierdzi sprzęt.

### Alternatywa cross-platform: goplay2

`goplay2` (Go) implementuje AirPlay 2 receiver natywnie (bez NQPTP), działa na macOS i Linuksie, claim ~1 ms PTP accuracy. Output domyślnie do PulseAudio (Linux) lub portaudio (macOS) — nie ma natywnego pipe PCM. Workaround na macOS: routing przez BlackHole / Loopback do procesu czytającego. Aktywność repo i wsparcie dla grupy z Samsungiem nieudokumentowane. [unverified — single source] [Source](https://github.com/openairplay/goplay2)

**Jeśli koniecznie macOS bez Pi**: użyj rekomendacji #1 (natywny macOS AirPlay receiver). `goplay2` jako third-party już nie potrzebny.

---

## Praktyczna rekomendacja #3 (Windows): trudne, prawdopodobnie nie warto

Windows **nie** trzyma UDP 319/320 (PtpClient opt-in), więc na poziomie OS port nie blokuje. Ale:

- shairport-sync Cygwin builds = **AirPlay 1 only** (CYGWIN.md: "AirPlay 2 Not Supported"). AirPlay 1 nie da się dodać do AP2 multi-room grupy z Q990F.
- WSL2 + shairport-sync AirPlay 2: jeden user uruchomił, zrezygnował, wrócił do Cygwin AP1. Wymaga mirrored networking mode (`.wslconfig`), Hyper-V firewall rules przez PowerShell admin, privileged port bind w WSL2. Nie do bundle'owania w PyInstallerze, tylko advanced manual setup.
- goplay2: zero binary releases, Linux-native deps (PulseAudio, fdk-aac).
- Brak utrzymywanego natywnego open-source AirPlay 2 receivera na Windowsa.

**Pragmatyczne opcje na Windowsie:**
- Pi 4 (~250 zł) jako audio gateway, VSLZR czyta przez WebSocket
- WSL2 manual setup (advanced users tylko, dokumentacja zewnętrzna)
- WASAPI loopback na PC + puszczanie z aplikacji desktop Spotify zamiast Spotify Connect (tracisz soundbar jako primary speaker — soundbar wtedy musi być Bluetooth sinkiem z PC, kompromis na jakości audio)
- Zostać przy mikrofonie

## Kluczowe findings na pytania bazowe

### Czy Q990F sam coś wystawia?

Q990F to *receive-only* endpoint sieciowy: AirPlay 2, Google Cast, Spotify Connect, Tidal Connect, Roon Ready (RAAT), DLNA/UPnP **renderer** (nie source). Wszystkie te role akceptują strumień; żadna go nie wypuszcza. [Source](https://www.whathifi.com/tv-home-cinema/soundbars/samsung-hw-q990f)

Reverse-engineered WAM HTTP API (port 55001/56001) na starszych Samsungach ma `GetMusicInfo` (metadata), `SetUrlPlayback` (push), volume/mute — **nigdy** endpointu typu "GetCurrentAudioStream". [Source](https://github.com/bacl/WAM_API_DOC/blob/master/API_Methods.md)

Na Q990B (2022) i nowszych Samsungach (silne implikacje dla Q990F) porty 55001/56001 są **zamknięte** — został tylko port 8080, używany przez nowsze HA integracje, też wyłącznie control. [Source](https://community.smartthings.com/t/why-port-56001-is-not-accessible-on-q990b-soundbar/273872) [unverified — port 8080 audio capability nie potwierdzony specyficznie dla Q990F, ekstrapolacja z Q990B/D]

SmartThings cloud API: `media_player` device z capabilities `switch`, `audioVolume`, `mediaInputSource`, `audioMute` — żadnego audio data. [Source](https://github.com/PiotrMachowski/Home-Assistant-custom-components-SmartThings-Soundbar)

**Confidence: high** — wszystkie cztery niezależne źródła (Samsung official, WAM reverse-eng, HA integracje, SmartThings docs) zgadzają się.

### Czy Spotify Connect pozwala na drugi odbiornik?

Nie. Protokół Connect jest single-active-device per konto. Audio nie leci telefon→soundbar tylko *Spotify CDN→soundbar* po tym jak soundbar staje się "active" (cloud-to-device, nie phone-to-device). Spirc protocol ma tylko transfer (przekazanie kontroli), nie broadcast. [Source](https://developer.spotify.com/documentation/commercial-hardware/implementation/requirements/technical)

Drugi `librespot` z tym samym kontem — kradnie playback (issue #793 dokumentuje "playback for a second then revert" gdy oba walczą o active slot). [Source](https://github.com/librespot-org/librespot/issues/793)

"Group" workaround Sonos/Echo/HomePod wymaga aby grupa broadcastowała się jako **pojedynczy ZeroConf device** (indywidualne głośniki ukryte). Nie da się dodać arbitralnego PC do grupy z Samsungiem — to działa tylko within-ecosystem (Sonos+Sonos, Echo+Echo). [Source](https://developer.spotify.com/documentation/commercial-hardware/implementation/requirements/technical)

Spotify Jam: dzieli queue/control, nie audio — każdy uczestnik gra na swoim własnym active device. [Source](https://github.com/orgs/music-assistant/discussions/419)

**Confidence: high.**

### Czy AirPlay 2 multi-receiver działa?

Tak — to natywna funkcja AirPlay 2 (multi-room). PTP synchronizuje próbki ~1 ms accuracy między grupowanymi receiverami. Jeden z receiverów może być open-source (`shairport-sync`) i równolegle wystawia audio jako PCM do FIFO — bezpośrednio konsumowane przez FFT pipeline.

Blocker dla macOS: NQPTP daemon shairport-sync chce ekskluzywne UDP 319/320, macOS odmawia. Stąd Pi jako host. shairport-sync `AIRPLAY2.md` explicitly to potwierdza i nie podaje workaroundu. [Source](https://github.com/mikebrady/shairport-sync/blob/master/AIRPLAY2.md)

**Confidence: high** dla blockera macOS; **medium** dla "grouping z Samsungiem działa bezproblemowo" — test empiryczny zalecany.

### Czy pasywny sniffing pomaga?

Nie. AirPlay 2 RTP payloads zaszyfrowane ChaCha20-Poly1305 AEAD; klucze ustanawiane podczas pairingu, nie transmitowane w późniejszych frame'ach. Wireshark widzi nagłówki/metadata, nie audio. [Source](https://emanuelecozzi.net/docs/airplay2/rtp)

Spotify Connect: stream idzie **z chmury Spotify bezpośrednio do soundbara** po TLS 1.2+ (telefon to tylko remote control). [Source](https://developer.spotify.com/blog/2024-10-01-outdated-tls-protocol-deprecation) Nie ma plaintext audio w LAN do "podsłuchania".

**Confidence: high.**

### Czy Spotify Web API to opłacalny fallback?

Praktycznie nie dla beat-sync visualizera.

Od **2024-11-27** Spotify usunął `audio-analysis`, `audio-features`, `recommendations`, `related-artists` dla **nowych aplikacji**. To były endpointy z pre-computed beat grid / segments / loudness — jedyny sposób żeby bez live audio sterować światłami zsynchronizowanymi z muzyką. [Source](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)

Extended access (zachowuje stare endpointy) wymaga od 2025-04-15 ≥250k MAU. Niedostępne dla solo projektu. [Source](https://developer.spotify.com/blog/2025-04-15-updating-the-criteria-for-web-api-extended-access)

`/v1/me/player/currently-playing` nadal działa ale `progress_ms` jest stale ~500-1000 ms, plus rate limity ~5-10 calls/30s na sensitive endpointy = realna latencja obserwacji ~550 ms+. Za grube na flash-na-beat. Nadaje się tylko do: track title/artist w UI, detekcja zmiany utworu, status playback. [Source](https://developer.spotify.com/documentation/web-api/concepts/rate-limits)

**Confidence: high** dla deprecation faktu; **medium** dla dokładnych rate limitów (Spotify ich oficjalnie nie publikuje per-endpoint).

## Konflikty

Brak istotnych konfliktów między źródłami. Wszystkie pięć podpytań zwraca spójny obraz: **AirPlay 2 multi-receiver to jedyna realna ścieżka; pozostałe drogi są zamknięte przez protokół, szyfrowanie albo politykę API.**

Jedyny mały rozjazd: niektóre HA integracje twierdzą że dostają `GetMusicInfo`-equivalent metadata przez port 8080 na nowszych Samsungach, ale spójnie nikt nie pokazuje audio data. To nie jest sprzeczność, raczej różne poziomy detalu.

## Caveats

- **Native macOS AirPlay receiver pominięty w pierwotnym researchu.** Workerzy zostali zaframowani na third-party receiverach (shairport-sync, goplay2, openairplay/airplay2-receiver) i nie postawili pytania "czy macOS ma to wbudowane?". Anchoring na source guidance + brak red-team passu (skipped, bo run był `deep` nie `exhaustive`) = miss. Praktyczna lekcja: kiedy worker wraca z werdyktem "X niemożliwe na platformie Y", to trigger do dodatkowego workera szukającego alternatyw poza pierwotnym source guidance, nie sygnał do zamknięcia tematu.
- **Q990F-specific port scan nie został niezależnie zweryfikowany** w publicznych źródłach. Findings o porcie 8080 / zamkniętym 55001/56001 są ekstrapolowane z Q990B/D. Pierwszy krok w testach: `nmap -p 1-65535` na Q990F i sprawdzić czy nie ma czegoś nowego/nieudokumentowanego.
- **shairport-sync + Samsung grouping** w iOS nie ma case-study którego znalazłem. Możliwe że iOS ukrywa shairport z Control Center kiedy wybiera "tylko certyfikowane AP2". Test empiryczny wymagany.
- **goplay2 macOS path** to single-source claim (README projektu). Nie ma threadów dokumentujących "działa z Samsungiem na macOS" w 2025-2026.
- **Apple Music → AirPlay 2 lossy downsample**: dla third-party receiverów (jak shairport albo nawet Q990F) Apple wysyła AAC 256 — Q990F nie dostanie lossless audio przez AirPlay z Apple Music, niezależnie od ustawień. Dla Spotify→AirPlay przepuszczenie i tak jest stratne (transcode).
- **librespot stabilność długoterminowa**: Spotify okresowo łamie nieoficjalnych klientów. [unverified — single source] Nie kluczowe dla tego projektu (rekomendowana ścieżka to AirPlay), ale warto zapamiętać.

## Sources

1. [Samsung HW-Q990F review — What Hi-Fi](https://www.whathifi.com/tv-home-cinema/soundbars/samsung-hw-q990f)
2. [WAM_API_DOC — bacl/GitHub](https://github.com/bacl/WAM_API_DOC)
3. [WAM API Methods](https://github.com/bacl/WAM_API_DOC/blob/master/API_Methods.md)
4. [SmartThings: port 56001 closed on Q990B](https://community.smartthings.com/t/why-port-56001-is-not-accessible-on-q990b-soundbar/273872)
5. [Samsung Soundbar Local — HA Community (port 8080)](https://community.home-assistant.io/t/samsung-soundbar-local/884397)
6. [PiotrMachowski SmartThings Soundbar HA integration](https://github.com/PiotrMachowski/Home-Assistant-custom-components-SmartThings-Soundbar)
7. [Roon Ready Devices — Samsung](https://roon.app/en/partners/206/samsung)
8. [Google Cast Intent to Join](https://developers.google.com/cast/docs/android_sender/intent_to_join)
9. [Spotify Connect Technical Requirements](https://developer.spotify.com/documentation/commercial-hardware/implementation/requirements/technical)
10. [Spotify Connect Basics](https://developer.spotify.com/documentation/commercial-hardware/implementation/guides/connect-basics)
11. [librespot GitHub](https://github.com/librespot-org/librespot)
12. [librespot spirc.rs source](https://github.com/librespot-org/librespot/blob/dev/connect/src/spirc.rs)
13. [librespot Audio Backends wiki](https://github.com/librespot-org/librespot/wiki/Audio-Backends)
14. [librespot issue #793 — multiple devices](https://github.com/librespot-org/librespot/issues/793)
15. [Music Assistant: Emulate Spotify Connect device](https://github.com/orgs/music-assistant/discussions/419)
16. [shairport-sync README](https://github.com/mikebrady/shairport-sync/blob/master/README.md)
17. [shairport-sync AIRPLAY2.md](https://github.com/mikebrady/shairport-sync/blob/master/AIRPLAY2.md)
18. [shairport-sync issue #1816 — simultaneous AP1+AP2](https://github.com/mikebrady/shairport-sync/issues/1816)
19. [shairport-sync issue #1437 — peppyalsa pipe](https://github.com/mikebrady/shairport-sync/issues/1437)
20. [goplay2 — AirPlay 2 receiver in Go](https://github.com/openairplay/goplay2)
21. [openairplay/airplay2-receiver — Python](https://github.com/openairplay/airplay2-receiver)
22. [Raspberry Pi AirPlay 2 setup gist](https://gist.github.com/maxonary/e51d367ff58403f21e4116855b31093b)
23. [AirPlay 2 stream lossy AAC — Darko.Audio](https://darko.audio/2023/10/apple-airplay-isnt-always-lossless-sometimes-its-lossy/)
24. [Spotify Web API Changes — Nov 27 2024](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)
25. [Spotify Extended API Access Criteria — Apr 2025](https://developer.spotify.com/blog/2025-04-15-updating-the-criteria-for-web-api-extended-access)
26. [Spotify Web API Rate Limits](https://developer.spotify.com/documentation/web-api/concepts/rate-limits)
27. [Spotify Currently Playing endpoint](https://developer.spotify.com/documentation/web-api/reference/get-the-users-currently-playing-track)
28. [Spotipy Issue #1173 — Audio Features 403](https://github.com/spotipy-dev/spotipy/issues/1173)
29. [AirPlay 2 RTP encryption — emanuelecozzi.net](https://emanuelecozzi.net/docs/airplay2/rtp)
30. [Apple AirPlay Capture — Weberblog](https://weberblog.net/apple-airplay-capture/)
31. [Spotify TLS 1.0/1.1 Deprecation — Oct 2024](https://developer.spotify.com/blog/2024-10-01-outdated-tls-protocol-deprecation)
32. [Spotify ZeroConf API](https://developer.spotify.com/documentation/commercial-hardware/implementation/guides/zeroconf)
33. [Spotify Connect explained — Android Police](https://www.androidpolice.com/spotify-connect-guide/)

<!-- METRICS:{"workers_initial":5,"workers_followup":0,"deep_fetcher_passes":0,"adversary_pass":false,"red_team_pass":false,"sources_unique":33,"depth_distribution":{"deep":3,"standard":1,"shallow":1},"mode":"web","constraints_used":true,"date":"2026-05-05","cost_tier":"deep","citation_precision":0.261,"citation_grader_note":"strict-verbatim grader; most UNSUPPORTED are valid synthesis from cited source"} -->
