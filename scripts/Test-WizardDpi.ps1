[CmdletBinding()]
param([string]$TaskId='20261106-000400-firstuse-dpi-6c8a')
$ErrorActionPreference='Stop'
if($PSVersionTable.PSEdition -ne 'Desktop'){throw 'Use Windows PowerShell 5.1 -STA.'}
if([Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA'){throw 'Use -STA.'}
if(-not $env:MOE4ALL_WIZARD_TEST_TMP -or -not $env:MOE4ALL_LAUNCHER_CACHE){throw 'Set task tmp and launcher-cache paths first.'}
if($TaskId -notmatch '^\d{8}-\d{6}-[A-Za-z0-9-]+$'){throw 'Invalid task ID'}
$workRoot='D:\AIinfr\infr\local-work'
function InTask([string]$Path,[string]$Category){$prefix=[IO.Path]::GetFullPath((Join-Path $workRoot "$Category\moe4all\$TaskId")).TrimEnd('\');$full=[IO.Path]::GetFullPath($Path).TrimEnd('\');return $full -eq $prefix -or $full.StartsWith($prefix+'\',[StringComparison]::OrdinalIgnoreCase)}
foreach($path in @($env:TEMP,$env:TMP,$env:MOE4ALL_WIZARD_TEST_TMP)){if(-not $path -or -not (InTask $path 'tmp')){throw 'All test/compiler temp paths must be inside this task tmp directory.'}}
if(-not (InTask $env:MOE4ALL_LAUNCHER_CACHE 'staging') -and -not (InTask $env:MOE4ALL_LAUNCHER_CACHE 'tmp')){throw 'Launcher test cache must be inside this task staging/tmp directory.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused-state';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$evidence=Join-Path 'D:\AIinfr\infr\local-work\evidence\moe4all' $TaskId
[void][IO.Directory]::CreateDirectory($evidence)
$state=Get-DashboardInitialState -Saved @{wizard_language='en';launch_mode='chat';model='';device='';setup_mode='aggressive'}
$schema=Get-DashboardSchema -State $state -DeviceOptions @([pscustomobject]@{value='';label='Automatic'})
$code=@'
using System;using System.Drawing;using System.Drawing.Imaging;using System.IO;using System.Reflection;using System.Runtime.InteropServices;using System.Text;using System.Windows.Forms;using MoE4All.Launcher;
public static class DpiChecks {
 [StructLayout(LayoutKind.Sequential)] struct Rect { public int L,T,R,B; }
 [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr h,int m,IntPtr w,IntPtr l);
 [DllImport("user32.dll")] static extern IntPtr GetCursor();
 [DllImport("user32.dll")] static extern IntPtr SetCursor(IntPtr c);
 [DllImport("user32.dll")] static extern IntPtr GetThreadDpiAwarenessContext();
 [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr c);
 [DllImport("user32.dll")] static extern IntPtr GetWindowDpiAwarenessContext(IntPtr h);
 [DllImport("user32.dll")] static extern bool AreDpiAwarenessContextsEqual(IntPtr a,IntPtr b);
 [DllImport("user32.dll")] static extern uint GetDpiForWindow(IntPtr h);
 static int checks;static readonly BindingFlags Hidden=BindingFlags.Instance|BindingFlags.NonPublic;
 static readonly StringBuilder report=new StringBuilder();
 static void Check(bool ok,string name){checks++;if(!ok)throw new Exception(name);}
 static Control Find(Control root,string name){var a=root.Controls.Find(name,true);if(a.Length!=1)throw new Exception("Missing "+name);return a[0];}
 static void Event(Control control,string name,EventArgs e){control.GetType().GetMethod(name,Hidden).Invoke(control,new object[]{e});}
 static IntPtr Packed(int x,int y){return new IntPtr(unchecked((int)(((uint)(ushort)y<<16)|(ushort)x)));}
 static int P(int n,int dpi){return (int)Math.Round(n*dpi/96.0);}
 static void Dpi(WizardWindow w,int dpi,int width,int height){
  Rect r=new Rect{L=-10000,T=-10000,R=-10000+P(width,dpi),B=-10000+P(height,dpi)};
  IntPtr memory=Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Rect)));
  try{Marshal.StructureToPtr(r,memory,false);SendMessage(w.Handle,0x02e0,new IntPtr(dpi|(dpi<<16)),memory);Application.DoEvents();}
  finally{Marshal.FreeHGlobal(memory);}
 }
 static Bitmap Image(Control c){var b=new Bitmap(c.Width,c.Height);c.DrawToBitmap(b,c.ClientRectangle);return b;}
 static bool Same(Bitmap a,Bitmap b){if(a.Size!=b.Size)return false;for(int y=0;y<a.Height;y++)for(int x=0;x<a.Width;x++)if(a.GetPixel(x,y)!=b.GetPixel(x,y))return false;return true;}
 static void Chrome(WizardWindow w,int dpi){
  int g=P(6,dpi),x=w.ClientSize.Width/2,y=w.ClientSize.Height/2;
  Point[] points={new Point(1,y),new Point(w.ClientSize.Width-2,y),new Point(x,1),new Point(x,w.ClientSize.Height-2),new Point(1,1),new Point(w.ClientSize.Width-2,1),new Point(1,w.ClientSize.Height-2),new Point(w.ClientSize.Width-2,w.ClientSize.Height-2)};
  int[] hits={10,11,12,15,13,14,16,17};Cursor[] cursors={Cursors.SizeWE,Cursors.SizeWE,Cursors.SizeNS,Cursors.SizeNS,Cursors.SizeNWSE,Cursors.SizeNESW,Cursors.SizeNESW,Cursors.SizeNWSE};
  IntPtr old=GetCursor();try{
   for(int i=0;i<points.Length;i++){Point screen=w.PointToScreen(points[i]);int hit=SendMessage(w.Handle,0x0084,IntPtr.Zero,Packed(screen.X,screen.Y)).ToInt32();Check(hit==hits[i],"Eight-way hit test at "+dpi+" hit "+hits[i]+" got "+hit);SendMessage(w.Handle,0x0020,w.Handle,Packed(hit,0x0200));Check(GetCursor()==cursors[i].Handle,"Eight-way cursor at "+dpi+" hit "+hit);}
  }finally{SetCursor(old);}
  Check(w.ContentHost.Left==g&&w.CaptionBar.Top==g,"Scaled resize band at "+dpi);
  Check(w.CaptionBar.Height==P(42,dpi),"Scaled caption height at "+dpi);
  foreach(string name in new[]{"CaptionMinimize","CaptionMaximize","CaptionClose"}){
   Control button=Find(w,name);Check(button.Width==P(46,dpi),"Caption target width "+name);
   Point center=button.PointToScreen(new Point(button.Width/2,button.Height/2));Check(SendMessage(w.Handle,0x0084,IntPtr.Zero,Packed(center.X,center.Y)).ToInt32()==1,"Caption mistaken for resize edge");
   using(var before=Image(button)){Event(button,"OnMouseEnter",EventArgs.Empty);using(var over=Image(button))Check(!Same(before,over),"Hover missing "+name);Event(button,"OnMouseLeave",EventArgs.Empty);using(var after=Image(button))Check(Same(before,after),"Hover residue "+name);}
  }
 }
 public static string Run(string state,string schema,string evidence){
  checks=0;Check(AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(),new IntPtr(-4)),"PowerShell UI thread is not PMv2");
  foreach(string theme in new[]{"light","dark"}){
   WizardTheme.SetPreference(theme);
   using(var w=new WizardWindow{Text="DPI fixture",Opacity=0,ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-10000,-10000),ClientSize=new Size(800,600)}){
    var sample=new Label{Name="Sample",Text="Native text Aa 012345",Font=new Font("Segoe UI",10),Bounds=new Rectangle(20,20,300,38)};w.ContentHost.Controls.Add(sample);
    var grid=new TableLayoutPanel{Top=90,Width=400,Height=120,ColumnCount=2,RowCount=2};grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,120));grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));grid.RowStyles.Add(new RowStyle(SizeType.Absolute,60));grid.RowStyles.Add(new RowStyle(SizeType.Percent,100));w.ContentHost.Controls.Add(grid);
    w.Show();Application.DoEvents();Check(AreDpiAwarenessContextsEqual(GetWindowDpiAwarenessContext(w.Handle),new IntPtr(-4)),"Native HWND is not PMv2");
    int liveDpi=(int)GetDpiForWindow(w.Handle);Check(Math.Abs(sample.Left-P(20,liveDpi))<=1,"Initial live monitor scaling");
    report.AppendLine("Live HWND context=PMv2 initial-monitor-dpi="+liveDpi);
    foreach(int dpi in new[]{96,120,144,192,144,120,96}){
     Dpi(w,dpi,800,600);Chrome(w,dpi);
     Check(Math.Abs(sample.Left-P(20,dpi))<=2&&Math.Abs(sample.Height-P(38,dpi))<=2,"Content bounds did not scale at "+dpi);
     Check(sample.Font.Unit==GraphicsUnit.Pixel&&Math.Abs(sample.Font.Size-10*dpi/72F)<0.1,"Native font did not rescale at "+dpi);
     Check(Math.Abs(grid.ColumnStyles[0].Width-P(120,dpi))<=2&&Math.Abs(grid.RowStyles[0].Height-P(60,dpi))<=2,"Absolute table styles did not scale at "+dpi+" col="+grid.ColumnStyles[0].Width+" row="+grid.RowStyles[0].Height);
     report.AppendLine(theme+" injected="+dpi+" native-monitor="+GetDpiForWindow(w.Handle)+" sample="+sample.Bounds+" font-px="+sample.Font.Size);
    }
   }
  }
  using(var ui=new NativeWizard(state,schema,"fixture",true,"") {Opacity=0,ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-10000,-10000)}){
   ui.Show();Application.DoEvents();
   foreach(int dpi in new[]{96,120,144,192,96}){
    Dpi(ui,dpi,1060,820);Control tile=Find(ui,"Task_chat");Check(Math.Abs(tile.Height-P(180,dpi))<=3,"Task tile did not scale at "+dpi);
    using(Bitmap image=Image(ui))image.Save(Path.Combine(evidence,"controlled-home-"+dpi+".png"),ImageFormat.Png);
    ui.SelectTask("chat");Application.DoEvents();Check(Find(ui,"TopBar").Height==P(48,dpi),"Page rebuild lost shell scale at "+dpi);
    Control field=Find(ui,"device");Check(field.Height>=P(32,dpi),"Rebuilt editor is too short at "+dpi);
    if(dpi==192){using(Bitmap image=Image(ui))image.Save(Path.Combine(evidence,"controlled-layout-200.png"),ImageFormat.Png);}
    // Rebuild the home page without invoking any engine or modal dialog.
    typeof(NativeWizard).GetField("screen",Hidden).SetValue(ui,"home");typeof(NativeWizard).GetMethod("Rebuild",Hidden).Invoke(ui,null);
   }
   Exception modalFailure=null;
   using(var timer=new System.Windows.Forms.Timer{Interval=30}){
    timer.Tick+=delegate{timer.Stop();var dialog=(WizardWindow)typeof(NativeWizard).GetField("settings",Hidden).GetValue(ui);
     try{foreach(int dpi in new[]{96,120,144,192,96}){
      Dpi(dialog,dpi,900,740);typeof(NativeWizard).GetMethod("FillSettings",Hidden).Invoke(ui,null);
      var tabs=(SettingsPages)typeof(NativeWizard).GetField("tabs",Hidden).GetValue(ui);
      foreach(var tab in tabs.TabPages){tabs.SelectedTab=tab;Application.DoEvents();Check(tab.Font.Unit==GraphicsUnit.Pixel,"Inactive settings page font not adapted");Check(tab.Controls.Count>0,"Empty settings fixture");Control group=tab.Controls[0];Check(group.Font.Size>=9*dpi/72F,"Inactive settings page used stale DPI");}
      Check(dialog.CaptionBar.Height==P(42,dpi),"Settings dialog chrome scale");
     }}catch(Exception ex){modalFailure=ex;}finally{if(dialog!=null)dialog.Close();}
    };timer.Start();ui.OpenSettings("reports","");
   }
   if(modalFailure!=null)throw modalFailure;
  }
  using(var child=new WizardWindow{ShowWindowButtons=false,Opacity=0,ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-10000,-10000)}){
   child.Show();Application.DoEvents();foreach(int dpi in new[]{96,120,144,192}){Dpi(child,dpi,700,500);Check(!Find(child,"CaptionClose").Visible,"Modal caption controls exposed");Check(child.CaptionBar.Height==P(42,dpi),"Modal caption DPI");}
  }
  // Never Show the runtime: its Shown handler starts the task. Exercise handles,
  // OnLoad and real WndProc scaling without starting any executable/session.
  using(var runtime=new RuntimeWindow("missing-test-fixture.exe",new string[0],evidence,"chat","","",true)){
   IntPtr handle=runtime.Handle;typeof(WizardWindow).GetMethod("OnLoad",Hidden).Invoke(runtime,new object[]{EventArgs.Empty});
   foreach(int dpi in new[]{96,120,144,192,96}){Dpi(runtime,dpi,1000,720);Control send=Find(runtime,"RuntimeSend");Check(send.Height>=send.Font.Height+P(6,dpi),"Runtime send text clipped at "+dpi);Check(send.Width>=P(90,dpi),"Runtime action width at "+dpi);}
  }
  File.WriteAllText(Path.Combine(evidence,"dpi-layout.log"),report.ToString());return checks+" checks passed; DPI values injected through WM_DPICHANGED, not physical multi-monitor acceptance.";
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'dpi-checks.cs'
[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativeWizard].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
$old=[DpiChecks]::SetThreadDpiAwarenessContext([IntPtr]::new(-1))
try{Initialize-DashboardRuntime;Write-Output ([DpiChecks]::Run(($state|ConvertTo-Json -Depth 25 -Compress),($schema|ConvertTo-Json -Depth 25 -Compress),$evidence))}
finally{[void][DpiChecks]::SetThreadDpiAwarenessContext($old)}
