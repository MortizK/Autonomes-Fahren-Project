Attribute VB_Name = "RGB_Fill"
Option Explicit

Public Sub FillRowsWithRgbColor()
    Dim sheet As Worksheet
    Dim redColumn As Long
    Dim greenColumn As Long
    Dim blueColumn As Long
    Dim lastRow As Long
    Dim lastColumn As Long
    Dim row As Long
    Dim red As Long
    Dim green As Long
    Dim blue As Long
    Dim color As Long

    Set sheet = ActiveSheet

    redColumn = FindHeaderColumn(sheet, "red")
    greenColumn = FindHeaderColumn(sheet, "green")
    blueColumn = FindHeaderColumn(sheet, "blue")

    If redColumn = 0 Or greenColumn = 0 Or blueColumn = 0 Then
        MsgBox "Die Spalten red, green und blue wurden nicht gefunden.", _
               vbExclamation
        Exit Sub
    End If

    lastRow = sheet.Cells(sheet.Rows.Count, redColumn).End(xlUp).row
    lastColumn = sheet.Cells(1, sheet.Columns.Count).End(xlToLeft).Column

    For row = 2 To lastRow
        If IsByteValue(sheet.Cells(row, redColumn).Value) _
                And IsByteValue(sheet.Cells(row, greenColumn).Value) _
                And IsByteValue(sheet.Cells(row, blueColumn).Value) Then
            red = CLng(sheet.Cells(row, redColumn).Value)
            green = CLng(sheet.Cells(row, greenColumn).Value)
            blue = CLng(sheet.Cells(row, blueColumn).Value)
            color = RGB(red, green, blue)

            sheet.Range(sheet.Cells(row, 1), sheet.Cells(row, lastColumn)).Interior.Color = color
        End If
    Next row

    MsgBox "Die RGB-Füllfarben wurden für " & (lastRow - 1) & _
           " Datenzeilen gesetzt.", vbInformation
End Sub

Private Function FindHeaderColumn(ByVal sheet As Worksheet, ByVal header As String) As Long
    Dim headerCell As Range

    Set headerCell = sheet.Rows(1).Find( _
        What:=header, _
        LookIn:=xlValues, _
        LookAt:=xlWhole, _
        MatchCase:=False)

    If Not headerCell Is Nothing Then
        FindHeaderColumn = headerCell.Column
    Else
        FindHeaderColumn = 0
    End If
End Function

Private Function IsByteValue(ByVal value As Variant) As Boolean
    IsByteValue = IsNumeric(value) _
        And CDbl(value) >= 0 _
        And CDbl(value) <= 255 _
        And CDbl(value) = Int(CDbl(value))
End Function
