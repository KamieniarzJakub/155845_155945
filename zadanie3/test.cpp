Struktury danych:
- Kandydaci[v] — lista K najbliższych sąsiadów wierzchołka v
- next[v], prev[v], inCycle[v] — reprezentacja aktualnego cyklu
- Position[v] — pozycja wierzchołka w chwilowej linearyzacji cyklu
- MoveType ∈ {NONE, INTRA_1, INTRA_2, ADD_NEXT, ADD_PREV, REMOVE}


FUNKCJA Solve(Instancja)
    Jeśli listy kandydackie nie są policzone
    lub ich rozmiar nie zgadza się z Instancja.n
    lub zmieniła się liczba kandydatów:
        PrecomputeCandidateMoves(Instancja, K)

    Sol = losowe rozwiązanie początkowe

    Jeśli Sol.cycle jest pusty:
        Oblicz statystyki Sol
        Zwróć Sol

    Zbuduj reprezentację cyklu:
        next[v], prev[v], inCycle[v]

    startVertex = pierwszy wierzchołek cyklu
    cycleSize = rozmiar cyklu
    Position = tablica rozmiaru Instancja.n wypełniona -1

    Improved = Prawda

    Dopóki Improved == Prawda:
        Improved = Fałsz

        BestDelta = 0
        BestType = NONE
        BestI = -1
        BestJorV = -1

        Cykl = linearyzacja aktualnego cyklu
        N = rozmiar(Cykl)

        Dla każdego i od 0 do N - 1:
            Position[Cykl[i]] = i

        // ruchy kandydackie
        Dla każdego i od 0 do N - 1:
            n1 = Cykl[i]
            n1_next = Cykl[(i + 1) MOD N]
            n1_prev = Cykl[(i - 1 + N) MOD N]

            Dla każdego n2 z Kandydaci[n1]:
                Jeśli inCycle[n2] == Prawda:
                    j = Position[n2]
                W przeciwnym wypadku:
                    j = -1

                Jeśli j != -1:
                    Jeśli i == j LUB i == (j + 1) MOD N LUB j == (i + 1) MOD N:
                        Kontynuuj

                    n2_next = Cykl[(j + 1) MOD N]
                    n2_prev = Cykl[(j - 1 + N) MOD N]

                    // INTRA_1
                    Delta1 =
                        (dist[n1][n1_next] + dist[n2_prev][n2]) -
                        (dist[n1][n2_prev] + dist[n1_next][n2])

                    Jeśli Delta1 > BestDelta:
                        zapamiętaj ruch INTRA_1

                    // INTRA_2
                    Delta2 =
                        (dist[n1_prev][n1] + dist[n2][n2_next]) -
                        (dist[n1_prev][n2] + dist[n1][n2_next])

                    Jeśli Delta2 > BestDelta:
                        zapamiętaj ruch INTRA_2

                W przeciwnym wypadku:
                    // ADD_NEXT
                    DeltaAddNext =
                        Profit[n2] - (dist[n1][n2] + dist[n2][n1_next] - dist[n1][n1_next])

                    Jeśli DeltaAddNext > BestDelta:
                        zapamiętaj ruch ADD_NEXT

                    // ADD_PREV
                    DeltaAddPrev =
                        Profit[n2] - (dist[n1_prev][n2] + dist[n2][n1] - dist[n1_prev][n1])

                    Jeśli DeltaAddPrev > BestDelta:
                        zapamiętaj ruch ADD_PREV

        // REMOVE
        Jeśli N > 2:
            Dla każdego idx od 0 do N - 1:
                v = Cykl[idx]
                prev_v = Cykl[(idx - 1 + N) MOD N]
                next_v = Cykl[(idx + 1) MOD N]

                DeltaRem =
                    (dist[prev_v][v] + dist[v][next_v] - dist[prev_v][next_v]) - Profit[v]

                Jeśli DeltaRem > BestDelta:
                    zapamiętaj ruch REMOVE

        Jeśli BestDelta > 0:
            Improved = Prawda

            Jeśli BestType == INTRA_1:
                ReverseSubpath(Cykl, BestI + 1, BestJorV - 1)
                Odbuduj next i prev z nowej kolejności
                startVertex = Cykl[0]

            Jeśli BestType == INTRA_2:
                ReverseSubpath(Cykl, BestI, BestJorV)
                Odbuduj next i prev z nowej kolejności
                startVertex = Cykl[0]

            Jeśli BestType == ADD_NEXT:
                Wstaw nowy wierzchołek za Cykl[BestI]
                cycleSize = cycleSize + 1

            Jeśli BestType == ADD_PREV:
                Wstaw nowy wierzchołek przed Cykl[BestI]
                cycleSize = cycleSize + 1
                Jeśli Cykl[BestI] == startVertex:
                    startVertex = nowy wierzchołek

            Jeśli BestType == REMOVE:
                Usuń wskazany wierzchołek z cyklu
                cycleSize = cycleSize - 1
                Jeśli usunięto startVertex:
                    startVertex = jego następnik

    Sol.cycle = MaterializeCycle(startVertex, next, cycleSize)
    Oblicz statystyki Sol
    Zwróć Sol


PROCEDURA PrecomputeCandidateMoves(Instancja, K)
    Dla każdego wierzchołka n1:
        distances = lista wszystkich par (dist[n1][n2], n2), gdzie n2 ≠ n1
        Posortuj distances rosnąco
        Kandydaci[n1] = pierwsze K wierzchołków z distances