nPopulacji <- 20
limitCzasowy <- Średni czas MSLS
czasStart <- PobierzAktualnyCzas()
populacja <- []
licznikIteracji <- 0

// 1. Inicjalizacja populacji (Minima lokalne)
DOPÓKI Rozmiar(populacja) < nPopulacji:
    solStart <- GenerujLosoweRozwiązanie()
    sol <- LocalSearchWithMoveList(solStart)
    JEŚLI sol jest unikalna w populacja:
        Dodaj sol do populacja

// 2. Pętla ewolucyjna (Steady-state)
DOPÓKI (PobierzAktualnyCzas() - czasStart) < limitCzasowy:
    // Selekcja rodziców
    rodzic1, rodzic2 <- WylosujDwaRóżne(populacja)

    // Rekombinacja i perturbacja
    dziecko <- Rekombinacja(rodzic1, rodzic2, opType)
    dziecko <- PerturbacjaMacro(dziecko) // Shaw Removal
    dziecko <- NaprawZRegret2(dziecko)

    // Doskonalenie
    JEŚLI useLS:
        dziecko <- LocalSearchWithMoveList(dziecko)
    W PRZECIWNYM RAZIE:
        PrzeliczStatystyki(dziecko)

    licznikIteracji <- licznikIteracji + 1

    // Zastępowanie najgorszego (Replacement)
    najgorszyIdx <- IndeksNajgorszego(populacja)
    JEŚLI FunkcjaCelu(dziecko) > FunkcjaCelu(populacja[najgorszyIdx]):
        JEŚLI dziecko jest unikalna w populacja:
            populacja[najgorszyIdx] <- dziecko

ZWRÓĆ NajlepszyZ(populacja)



FUNKCJA Rekombinacja(rodzic1, rodzic2, typ):
    JEŚLI typ == 1: // Wspólne krawędzie i wierzchołki
        krawędzie2 <- ZbiórKrawędzi(rodzic2)
        podścieżki <- WyodrębnijWspólneFragmenty(rodzic1, krawędzie2)
        ZWRÓĆ PołączLosowo(podścieżki)

    JEŚLI typ == 2: // Redukcja do części wspólnych
        v2 <- ZbiórWierzchołków(rodzic2)
        e2 <- ZbiórKrawędzi(rodzic2)
        podścieżki <- []
        DLA KAŻDEGO u, v w rodzic1:
            JEŚLI u należy do v2:
                Dodaj u do bieżącaPodścieżka
                JEŚLI krawędź(u, v) nie należy do e2:
                    ZamknijBieżącąPodścieżkę()
        ZWRÓĆ PołączLosowo(podścieżki)

    JEŚLI typ == 3: // Tylko wspólne wierzchołki
        dziecko <- []
        v2 <- ZbiórWierzchołków(rodzic2)
        DLA KAŻDEGO v w rodzic1:
            JEŚLI v należy do v2:
                Dodaj v do dziecko.cykl
        ZWRÓĆ dziecko


FUNKCJA PerturbacjaMacro(sol):
    n_cykl <- Rozmiar(sol.cykl)
    q <- LosujWartość(0.15, 0.45)
    doUsunięcia <- n_cykl * q
    
    Tablica keep[] <- {true, ..., true}
    liczbaSegmentów <- 3
    
    DLA s = 0 DO liczbaSegmentów - 1:
        start_v <- WylosujJeden(NieusunięteWierzchołkiCyklu)
        
        Lista podobieństwa <- ObliczPodobieństwoShaw(start_v, sol.cykl)
        PosortujRosnąco(podobieństwa)
        
        keep[start_v] <- false
        DLA i = 0 DO (doUsunięcia / liczbaSegmentów) - 2:
            v_podobny <- podobieństwa[i].v
            keep[v_podobny] <- false
            
    sol.cykl <- Przefiltruj(sol.cykl, keep == true)
    PrzeliczStatystyki(sol)
    ZWRÓĆ sol






FUNKCJA NaprawZRegret2(sol):
    n <- instancja.liczbaWierzchołków
    cel_rozmiar <- n / 2
    nieużywane <- PobierzWierzchołkiPozaCykl(sol)

    DOPÓKI Rozmiar(sol.cykl) < cel_rozmiar:
        NajlepszyWierzchołek <- null
        MaxRegret <- -1
        NajlepszaPozycja <- -1

        DLA KAŻDEGO v W nieużywane:
            // Obliczamy koszt wstawienia v w każde możliwe miejsce w cyklu
            koszty <- ObliczWszystkieKosztyWstawienia(sol.cykl, v) 
            PosortujRosnąco(koszty) // koszty[0] - najtańsze, koszty[1] - drugie najtańsze

            // Regret to różnica między drugim a pierwszym najlepszym wstawieniem
            // Uwzględniamy zysk wierzchołka: koszt_netto = koszt_krawędzi - zysk_v
            regret <- (koszty[1].wartość - v.zysk) - (koszty[0].wartość - v.zysk)

            JEŚLI regret > MaxRegret:
                MaxRegret <- regret
                NajlepszyWierzchołek <- v
                NajlepszaPozycja <- koszty[0].pozycja

        // Wstawiamy wierzchołek, który generuje największą "stratę" przy pominięciu
        WstawW Cykl(sol.cykl, NajlepszyWierzchołek, NajlepszaPozycja)
        UsuńZ Listy(nieużywane, NajlepszyWierzchołek)

    PrzeliczStatystyki(sol)
    ZWRÓĆ sol