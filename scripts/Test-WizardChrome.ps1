[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Set the project work root and task tmp directory first.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused'
$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
Initialize-DashboardRuntime
$code=@'
using System;using System.Drawing;using System.Reflection;using System.Runtime.InteropServices;using System.Windows.Forms;using MoE4All.Launcher;
public static class ChromeChecks {
 [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr h,int index);
 [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr h,int message,IntPtr w,IntPtr l);
 static int checks;
 static readonly BindingFlags Hidden=BindingFlags.Instance|BindingFlags.NonPublic;
 static void Check(bool ok,string reason){checks++;if(!ok)throw new Exception(reason);}
 static Control Find(Control root,string name){var list=root.Controls.Find(name,true);return list.Length==0?null:list[0];}
 static void Event(Control c,string name,object value){c.GetType().GetMethod(name,Hidden).Invoke(c,new[]{value});Application.DoEvents();}
 static Bitmap Image(Control c){var b=new Bitmap(c.Width,c.Height);c.DrawToBitmap(b,c.ClientRectangle);return b;}
 static bool Equal(Bitmap a,Bitmap b){if(a.Size!=b.Size)return false;for(int y=0;y<a.Height;y++)for(int x=0;x<a.Width;x++)if(a.GetPixel(x,y)!=b.GetPixel(x,y))return false;return true;}
 static void Mouse(Control c,string name,int x,int y){Event(c,name,new MouseEventArgs(MouseButtons.Left,1,x,y,0));}
 public static int Run(){checks=0;
  foreach(string theme in new[]{"dark","light"}){
   WizardTheme.SetPreference(theme);
   using(var form=new WizardWindow{Opacity=0,ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-5000,-5000),Size=new Size(800,600)}){
    form.Show();Application.DoEvents();form.RefreshTheme();
    Check((GetWindowLong(form.Handle,-16)&0x00c40000)==0,"Native frame remains");
    var max=Find(form,"CaptionMaximize");var min=Find(form,"CaptionMinimize");var close=Find(form,"CaptionClose");
    foreach(var c in new[]{max,min,close}){
     Check(c!=null&&c.Visible&&!(c is Button),"Caption still uses native Button painting");
     using(var before=Image(c)){Event(c,"OnMouseEnter",EventArgs.Empty);using(var over=Image(c))Check(!Equal(before,over),"Caption hover missing");Event(c,"OnMouseLeave",EventArgs.Empty);using(var after=Image(c))Check(Equal(before,after),"Caption hover leaves a paint residue");}
    }
    Mouse(max,"OnMouseDown",8,8);Mouse(max,"OnMouseUp",8,8);
    Check(form.WindowState==FormWindowState.Maximized,"Mouse maximize failed or toggled twice");
    Mouse(max,"OnMouseDown",8,8);Mouse(max,"OnMouseUp",8,8);
    Check(form.WindowState==FormWindowState.Normal,"Mouse restore failed");
    Mouse(min,"OnMouseDown",8,8);Mouse(min,"OnMouseUp",8,8);
    Check(form.WindowState==FormWindowState.Minimized,"Mouse minimize failed");
    form.WindowState=FormWindowState.Normal;Application.DoEvents();
    var page=new WizardScrollView{Dock=DockStyle.Fill,AutoScrollMinSize=new Size(0,1400)};form.ContentHost.Controls.Add(page);
    var choice=new WizardComboBox{DropDownStyle=ComboBoxStyle.DropDownList,Location=new Point(20,20),Width=220};choice.Items.Add("First");choice.Items.Add("Second");choice.SelectedIndex=1;page.Controls.Add(choice);form.RefreshTheme();Application.DoEvents();
    SendMessage(choice.Handle,0x020a,new IntPtr(-120<<16),IntPtr.Zero);
    Check(choice.SelectedIndex==1&&page.ScrollOffset>0,"Wheel changes choice instead of scrolling");
    Check((GetWindowLong(page.Handle,-16)&0x00300000)==0,"Native page scrollbar remains");
    var track=(WizardScrollTrack)typeof(WizardScrollView).GetField("track",Hidden).GetValue(page);var thumb=(Rectangle)typeof(WizardScrollTrack).GetMethod("Thumb",Hidden).Invoke(track,null);int offset=page.ScrollOffset;
    Mouse(track,"OnMouseDown",thumb.X+2,thumb.Y+2);Mouse(track,"OnMouseMove",thumb.X+2,thumb.Y+70);Mouse(track,"OnMouseUp",thumb.X+2,thumb.Y+70);
    Check(page.ScrollOffset>offset,"Managed scrollbar thumb cannot drag");
    var editor=new WizardComboBox{Location=new Point(20,220),Width=220,Text="keep this path"};page.Controls.Add(editor);Application.DoEvents();int previous=page.ScrollOffset;SendMessage(editor.Controls[0].Handle,0x020a,new IntPtr(-120<<16),IntPtr.Zero);
    Check(editor.Text=="keep this path"&&page.ScrollOffset>previous,"Wheel over editable choice does not scroll");
    bool closed=false;form.FormClosed+=delegate{closed=true;};((IButtonControl)close).PerformClick();Application.DoEvents();Check(closed,"Close action did not enter Form lifecycle");
   }
   using(var child=new WizardWindow{ShowWindowButtons=false,Opacity=0,ShowInTaskbar=false}){child.Show();Application.DoEvents();foreach(string name in new[]{"CaptionMinimize","CaptionMaximize","CaptionClose"})Check(!Find(child,name).Visible,"Child exposes a caption button");child.Close();}
  }
  return checks;
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'chrome-checks.cs'
[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativeWizard].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
Write-Output ('Chrome and scrolling: '+[ChromeChecks]::Run()+' checks passed. Own invisible forms only; no engine/GPU/network/user-state access.')
