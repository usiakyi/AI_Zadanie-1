
import numpy as np
import csv
import tempfile
import os

# ==========================================
# 1. DANE POSTACI
# ==========================================

NAZWY = np.array([
    "Wojownik", "Mag", "Łucznik", "Paladyn",
    "Łotrzyk", "Berserker", "Druid", "Strażnik"
])

STATYSTYKI = np.array([
    [120, 28, 18, 9],
    [75, 44, 6, 13],
    [95, 34, 10, 17],
    [140, 25, 22, 7],
    [80, 39, 8, 21],
    [110, 42, 7, 12],
    [105, 26, 15, 14],
    [155, 18, 26, 5]
], dtype=float)

WAGI = np.array([0.20, 0.40, 0.25, 0.15])

KOLUMNY = ("HP", "atak", "obrona", "szybkosc")


# ==========================================
# 2. WALIDACJA DANYCH
# ==========================================

def waliduj(nazwy, dane):

    if not isinstance(dane, np.ndarray):
        raise ValueError("Dane statystyczne muszą być tablicą NumPy.")

    if dane.ndim != 2:
        raise ValueError("Macierz statystyk musi być dwuwymiarowa.")

    if dane.shape[1] != 4:
        raise ValueError("Macierz musi mieć dokładnie 4 kolumny.")

    if dane.shape[0] != len(nazwy):
        raise ValueError("Liczba wierszy musi odpowiadać liczbie nazw postaci.")

    if len(nazwy) < 2:
        raise ValueError("Muszą istnieć co najmniej dwie postacie.")

    if len(np.unique(nazwy)) != len(nazwy):
        raise ValueError("Nazwy postaci muszą być unikatowe.")

    try:
        if not np.all(np.isfinite(dane)):
            raise ValueError("Statystyki muszą być liczbami skończonymi.")
    except TypeError:
        raise ValueError("Statystyki muszą zawierać wartości liczbowe.")

    if np.any(dane < 0):
        raise ValueError("Statystyki nie mogą być ujemne.")

    return True


# ==========================================
# 3. NORMALIZACJA I OBLICZANIE SIŁY
# ==========================================

def normalizuj(dane):

    minimum = np.min(dane, axis=0)
    zakres = np.ptp(dane, axis=0)

    znormalizowane = np.divide(
        dane - minimum,
        zakres,
        out=np.zeros_like(dane, dtype=float),
        where=zakres != 0
    )

    return znormalizowane


def oblicz_sile(dane):

    dane_norm = normalizuj(dane)
    sila = dane_norm @ WAGI

    return sila


# ==========================================
# 4. OKREŚLANIE RÓL
# ==========================================

def okresl_role(dane):

    szturmowiec = (dane[:, 1] >= 35) & (dane[:, 3] >= 12)

    obronca = (dane[:, 0] >= 120) & (dane[:, 2] >= 18)

    role = np.select(
        [szturmowiec, obronca],
        ["Szturmowiec", "Obrońca"],
        default="Wszechstronny"
    )

    return role


# ==========================================
# 5. RANKING POSTACI
# ==========================================

def ranking_postaci(nazwy, sila):

    indeksy = np.argsort(-sila, kind="stable")

    return nazwy[indeksy], sila[indeksy]


# ==========================================
# 6. MACIERZ POJEDYNKÓW
# ==========================================

def macierz_pojedynkow(sila):

    sila = np.asarray(sila, dtype=float)

    if sila.ndim != 1:
        raise ValueError("Siła musi być tablicą jednowymiarową.")

    if not np.all(np.isfinite(sila)):
        raise ValueError("Wartości siły muszą być skończone.")

    roznice = sila[:, None] - sila[None, :]

    prawdopodobienstwa = 1 / (1 + np.exp(-8 * roznice))

    np.fill_diagonal(prawdopodobienstwa, 0.5)

    return prawdopodobienstwa


# ==========================================
# 7. ŚREDNIE SZANSE WYGRANEJ
# ==========================================

def srednie_szanse(macierz):

    if not isinstance(macierz, np.ndarray):
        raise ValueError("Macierz musi być tablicą NumPy.")

    if macierz.ndim != 2:
        raise ValueError("Macierz musi być dwuwymiarowa.")

    if macierz.shape[0] != macierz.shape[1]:
        raise ValueError("Macierz musi być kwadratowa.")

    n = macierz.shape[0]

    if n < 2:
        raise ValueError("Potrzebne są co najmniej dwie postacie.")

    maska = ~np.eye(n, dtype=bool)

    suma = np.sum(np.where(maska, macierz, 0), axis=1)

    srednie = suma / (n - 1)

    return srednie


# ==========================================
# 8. FILTROWANIE POSTACI
# ==========================================

def filtruj_postacie(nazwy, statystyki, sila):

    maska = (
        (statystyki[:, 0] >= 100) &
        (statystyki[:, 3] >= 10) &
        (sila >= 0.45)
    )

    wybrane_postacie = nazwy[maska]

    print("\n=== POSTACIE SPEŁNIAJĄCE WYMAGANIA ===")

    if len(wybrane_postacie) > 0:
        for nazwa in wybrane_postacie:
            print(nazwa)
    else:
        print("Brak")

    return wybrane_postacie


# ==========================================
# 9. STATYSTYKI CAŁEJ GRUPY
# ==========================================

def statystyki_grupy(dane, kolumny):

    print("\n=== STATYSTYKI CAŁEJ GRUPY ===")

    print(
        f"{'Statystyka':<10} "
        f"{'Minimum':>10} "
        f"{'Średnia':>10} "
        f"{'Maksimum':>10}"
    )

    for i in range(len(kolumny)):

        minimum = np.min(dane[:, i])
        srednia = np.mean(dane[:, i])
        maksimum = np.max(dane[:, i])

        print(
            f"{kolumny[i]:<10} "
            f"{minimum:>10.1f} "
            f"{srednia:>10.1f} "
            f"{maksimum:>10.1f}"
        )


# ==========================================
# 10. EKSPORT DO CSV
# ==========================================

def zapisz_csv(sciezka, nazwy, role, sila,
               srednie_szanse, ranking):

    nazwy_rankingu = ranking[0]

    indeksy = np.array([
        np.where(nazwy == nazwa)[0][0]
        for nazwa in nazwy_rankingu
    ])

    with open(
        sciezka,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as plik:

        writer = csv.writer(plik, delimiter=";")

        writer.writerow([
            "Miejsce",
            "Postać",
            "Rola",
            "Siła",
            "Średnia szansa wygranej"
        ])

        for miejsce, indeks in enumerate(indeksy, start=1):

            writer.writerow([
                miejsce,
                nazwy[indeks],
                role[indeks],
                f"{sila[indeks]:.4f}",
                f"{srednie_szanse[indeks] * 100:.2f}%"
            ])

    print(f"\nRaport CSV zapisano w pliku: {sciezka}")

def test_zapis_csv():

    print("\n=== TEST EKSPORTU CSV ===")

    # Przygotowanie danych do raportu
    sila_testowa = oblicz_sile(STATYSTYKI)
    role_testowe = okresl_role(STATYSTYKI)
    ranking_testowy = ranking_postaci(NAZWY, sila_testowa)

    macierz_testowa = macierz_pojedynkow(sila_testowa)
    srednie_testowe = srednie_szanse(macierz_testowa)

    # Utworzenie pliku tymczasowego
    with tempfile.TemporaryDirectory() as katalog:

        sciezka = os.path.join(katalog, "test_raport.csv")

        # Zapis raportu
        zapisz_csv(
            sciezka,
            NAZWY,
            role_testowe,
            sila_testowa,
            srednie_testowe,
            ranking_testowy
        )

        # Sprawdzenie kodowania UTF-8-SIG (BOM)
        with open(sciezka, "rb") as plik:
            zawartosc_bajtowa = plik.read()

        assert zawartosc_bajtowa.startswith(b"\xef\xbb\xbf"), (
            "Plik nie zawiera znacznika BOM kodowania UTF-8-SIG."
        )

        # Odczyt raportu z zachowaniem polskich znaków
        with open(
            sciezka,
            mode="r",
            newline="",
            encoding="utf-8-sig"
        ) as plik:

            czytnik = csv.reader(plik, delimiter=";")
            rekordy = list(czytnik)

        # Test 1: poprawny nagłówek
        oczekiwany_naglowek = [
            "Miejsce",
            "Postać",
            "Rola",
            "Siła",
            "Średnia szansa wygranej"
        ]

        assert rekordy[0] == oczekiwany_naglowek, (
            "Niepoprawny nagłówek CSV."
        )

        print("[OK] Poprawny nagłówek CSV")

        # Test 2: liczba rekordów
        assert len(rekordy) - 1 == 8, (
            "Raport powinien zawierać dokładnie 8 rekordów."
        )

        print("[OK] Raport zawiera 8 rekordów")

        # Test 3: separator średnikowy
        assert all(len(rekord) == 5 for rekord in rekordy), (
            "Nie wszystkie rekordy zawierają 5 kolumn."
        )

        pierwsza_linia = zawartosc_bajtowa.decode(
            "utf-8-sig"
        ).splitlines()[0]

        assert ";" in pierwsza_linia, (
            "Nie znaleziono separatora średnikowego."
        )

        print("[OK] Poprawny separator średnikowy")

        # Test 4: poprawne polskie znaki
        wszystkie_dane = [
            pole
            for rekord in rekordy
            for pole in rekord
        ]

        assert "Łucznik" in wszystkie_dane
        assert "Strażnik" in wszystkie_dane
        assert "Obrońca" in wszystkie_dane

        print("[OK] Poprawne kodowanie polskich znaków")

    print("Wszystkie testy eksportu CSV zakończone powodzeniem.")


# ==========================================
# 11. TESTY AUTOMATYCZNE
# ==========================================

def testy():

    print("\n=== TESTY AUTOMATYCZNE ===")

    # Test 1: normalizacja Min-Max

    dane_testowe = np.array([
        [10, 5],
        [20, 5],
        [30, 5]
    ], dtype=float)

    wynik = normalizuj(dane_testowe)

    oczekiwany = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0]
    ])

    assert np.allclose(wynik, oczekiwany)

    print("[OK] Test normalizacji Min-Max")

    # Test 2: obsługa stałej kolumny

    assert np.all(wynik[:, 1] == 0)

    print("[OK] Test stałej kolumny")

    # Test 3: macierz pojedynków

    sila_testowa = np.array([0.2, 0.5, 0.8])

    macierz = macierz_pojedynkow(sila_testowa)

    assert macierz.shape == (3, 3)

    assert np.allclose(np.diag(macierz), 0.5)

    assert np.allclose(macierz + macierz.T, 1.0)

    assert np.all((macierz >= 0) & (macierz <= 1))

    print("[OK] Test macierzy pojedynków")

    print("Wszystkie testy zakończone powodzeniem.")
    # Testy poprawności walidacji danych
    test_walidacji()
    # Test poprawności eksportu CSV
    test_zapis_csv()
# ==========================================
# 12. FUNKCJA GŁÓWNA
# ==========================================


def sprawdz_bledne_dane(opis, nazwy, dane, oczekiwany_komunikat):
    """Sprawdza, czy walidacja zgłasza odpowiedni ValueError."""

    try:
        waliduj(nazwy, dane)

    except ValueError as blad:
        assert oczekiwany_komunikat in str(blad), (
            f"{opis}: nieoczekiwany komunikat błędu: {blad}"
        )
        print(f"[OK] {opis}")

    else:
        raise AssertionError(
            f"{opis}: oczekiwano wyjątku ValueError, ale go nie zgłoszono."
        )


def test_walidacji():

    print("\n=== TESTY WALIDACJI ===")

    # Test 1: zduplikowane nazwy postaci

    nazwy_zduplikowane = NAZWY.copy()
    nazwy_zduplikowane[1] = nazwy_zduplikowane[0]

    sprawdz_bledne_dane(
        "Zduplikowane nazwy",
        nazwy_zduplikowane,
        STATYSTYKI,
        "Nazwy postaci muszą być unikatowe."
    )

    # Test 2: ujemne wartości statystyk

    dane_ujemne = STATYSTYKI.copy()
    dane_ujemne[0, 0] = -10

    sprawdz_bledne_dane(
        "Ujemne statystyki",
        NAZWY,
        dane_ujemne,
        "Statystyki nie mogą być ujemne."
    )

    # Test 3: wartości NaN

    dane_nan = STATYSTYKI.copy()
    dane_nan[0, 0] = np.nan

    sprawdz_bledne_dane(
        "Wartości NaN",
        NAZWY,
        dane_nan,
        "Statystyki muszą być liczbami skończonymi."
    )

    # Test 4: nieprawidłowy wymiar tablicy

    dane_zly_wymiar = STATYSTYKI.flatten()

    sprawdz_bledne_dane(
        "Nieprawidłowy wymiar tablicy",
        NAZWY,
        dane_zly_wymiar,
        "Macierz statystyk musi być dwuwymiarowa."
    )

    print("Wszystkie testy walidacji zakończone powodzeniem.")
# Test 5: wartości nieskończone

dane_inf = STATYSTYKI.copy()
dane_inf[0, 0] = np.inf

sprawdz_bledne_dane(
    "Wartości nieskończone",
    NAZWY,
    dane_inf,
    "Statystyki muszą być liczbami skończonymi."
)


def main():
    # Walidacja danych
    waliduj(NAZWY, STATYSTYKI)

    # Obliczanie sily postaci
    sila = oblicz_sile(STATYSTYKI)

    # Okreslanie rol
    role = okresl_role(STATYSTYKI)

    # Ranking postaci
    ranking = ranking_postaci(NAZWY, sila)

    # Macierz pojedynkow
    macierz = macierz_pojedynkow(sila)

    # Srednie szanse wygranej
    srednie = srednie_szanse(macierz)

    print("\n=== RANKING POSTACI ===")

    for i, nazwa in enumerate(ranking[0], start=1):
        print(f"{i}. {nazwa}: {ranking[1][i-1]:.4f}")

    # Statystyki grupy
    statystyki_grupy(STATYSTYKI, KOLUMNY)

    # Filtrowanie postaci
    filtruj_postacie(NAZWY, STATYSTYKI, sila)

    # Eksport CSV
    zapisz_csv(
        "raport.csv",
        NAZWY,
        role,
        sila,
        srednie,
        ranking
    )

    # Testy automatyczne
    testy()


if __name__ == "__main__":
    main()
