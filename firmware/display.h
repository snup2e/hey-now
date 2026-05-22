// Display interface for Path 1 — 2.0" SPI LCD ("지난 역 / 현재 역").
//
// The concrete LCD driver (ILI9341-class panel) is wired up during
// firmware integration; this header is the contract app_demo.c relies on.
#ifndef DISPLAY_H
#define DISPLAY_H

#include "app_path1.h"

// Initialize the LCD panel.
void lcd_init(void);

// Show the route state: previous station / current station.
// Uses STATION_NAMES[]; handles the -1 (none) case.
void lcd_show_route(const Path1State *st);

// Show two free-form text lines (startup / status messages).
void lcd_show_message(const char *line1, const char *line2);

#endif  // DISPLAY_H
