[CmdletBinding()]
param(
    [string]$SvgPath = '',
    [Parameter(Mandatory=$true)][string]$OutputPath
)
# Developer-only rasterization of the website favicon using inbox .NET/WPF.
# The checked-in ICO is shipped; users need no SVG renderer or new dependency.
$ErrorActionPreference='Stop'
if (-not $SvgPath) { $SvgPath=Join-Path $PSScriptRoot 'moe4all-icon.svg' }
Add-Type -AssemblyName PresentationCore,WindowsBase
[Windows.Media.RenderOptions]::ProcessRenderMode=[Windows.Interop.RenderMode]::SoftwareOnly
[xml]$svg=[IO.File]::ReadAllText($SvgPath)
$size=[double]$svg.svg.width
$background=[Windows.Media.BrushConverter]::new().ConvertFromString([string]$svg.svg.rect.fill)
$stroke=[Windows.Media.BrushConverter]::new().ConvertFromString([string]$svg.svg.path.stroke)
$geometry=[Windows.Media.Geometry]::Parse([string]$svg.svg.path.d)
$pen=[Windows.Media.Pen]::new($stroke,[double]$svg.svg.path.'stroke-width')
$pen.StartLineCap=[Windows.Media.PenLineCap]::Round;$pen.EndLineCap=$pen.StartLineCap;$pen.LineJoin=[Windows.Media.PenLineJoin]::Round
$frames=@()
foreach($pixels in @(16,24,32,48,64,256)){
    $visual=[Windows.Media.DrawingVisual]::new();$drawing=$visual.RenderOpen()
    try {
        $drawing.PushTransform([Windows.Media.ScaleTransform]::new($pixels/$size,$pixels/$size))
        $drawing.DrawRoundedRectangle($background,$null,[Windows.Rect]::new(0,0,$size,$size),[double]$svg.svg.rect.rx,[double]$svg.svg.rect.rx)
        $drawing.DrawGeometry($null,$pen,$geometry);$drawing.Pop()
    } finally {$drawing.Close()}
    $bitmap=[Windows.Media.Imaging.RenderTargetBitmap]::new($pixels,$pixels,96,96,[Windows.Media.PixelFormats]::Pbgra32);$bitmap.Render($visual)
    $encoder=[Windows.Media.Imaging.PngBitmapEncoder]::new();$encoder.Frames.Add([Windows.Media.Imaging.BitmapFrame]::Create($bitmap))
    $stream=[IO.MemoryStream]::new();try{$encoder.Save($stream);$frames+=@{size=$pixels;data=$stream.ToArray()}}finally{$stream.Dispose()}
}
$OutputPath=[IO.Path]::GetFullPath($OutputPath);[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($OutputPath))
$stream=[IO.File]::Create($OutputPath);$writer=[IO.BinaryWriter]::new($stream)
try{
    $writer.Write([uint16]0);$writer.Write([uint16]1);$writer.Write([uint16]$frames.Count);$offset=6+16*$frames.Count
    foreach($frame in $frames){$dimension=if($frame.size-eq256){0}else{$frame.size};$writer.Write([byte]$dimension);$writer.Write([byte]$dimension);$writer.Write([uint16]0);$writer.Write([uint16]1);$writer.Write([uint16]32);$writer.Write([uint32]$frame.data.Length);$writer.Write([uint32]$offset);$offset+=$frame.data.Length}
    foreach($frame in $frames){$writer.Write([byte[]]$frame.data)}
}finally{$writer.Dispose();$stream.Dispose()}
Write-Output $OutputPath
