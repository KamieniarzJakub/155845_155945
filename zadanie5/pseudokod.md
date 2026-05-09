PROCEDURA PobierzWierzcholki(Rozwiązanie):
    ZbiórW = pusty zbiór
    DLA KAŻDEGO wierzchołka W w Rozwiązanie.cykl:
        Dodaj W do ZbiórW
    ZWRÓĆ ZbiórW

PROCEDURA PobierzKrawedzie(Rozwiązanie):
    ZbiórK = pusty zbiór
    N = długość cyklu
    DLA i OD 0 DO N-1:
        U = Rozwiązanie.cykl[i]
        V = Rozwiązanie.cykl[(i + 1) MOD N]
        // Normalizacja: mniejszy indeks zawsze pierwszy
        Krawędź = PARA(MIN(U, V), MAX(U, V))
        Dodaj Krawędź do ZbiórK
    ZWRÓĆ ZbiórK

PROCEDURA PoliczWspolneElementy(ZbiórA, ZbiórB):
    Licznik = 0
    DLA KAŻDEGO elementu E w ZbiórA:
        JEŚLI E znajduje się w ZbiórB:
            Licznik = Licznik + 1
    ZWRÓĆ Licznik

PROCEDURA ObliczKorelacje(Zmienna_X, Zmienna_Y):
    N = liczba próbek (1000)
    ŚredniaX = Suma(Zmienna_X) / N
    ŚredniaY = Suma(Zmienna_Y) / N
    
    Licznik = 0
    MianownikX = 0
    MianownikY = 0
    
    DLA i OD 0 DO N-1:
        Dx = Zmienna_X[i] - ŚredniaX
        Dy = Zmienna_Y[i] - ŚredniaY
        Licznik = Licznik + (Dx * Dy)
        MianownikX = MianownikX + (Dx^2)
        MianownikY = MianownikY + (Dy^2)
        
    ZWRÓĆ Licznik / PIERWIASTEK(MianownikX * MianownikY)