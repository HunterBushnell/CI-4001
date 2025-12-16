#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _gap_reg(void);
extern void _k_rtm_reg(void);
extern void _k_wb_reg(void);
extern void _leak_reg(void);
extern void _na_rtm_reg(void);
extern void _na_wb_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"./gap.mod\"");
    fprintf(stderr, " \"./k_rtm.mod\"");
    fprintf(stderr, " \"./k_wb.mod\"");
    fprintf(stderr, " \"./leak.mod\"");
    fprintf(stderr, " \"./na_rtm.mod\"");
    fprintf(stderr, " \"./na_wb.mod\"");
    fprintf(stderr, " \"./vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _gap_reg();
  _k_rtm_reg();
  _k_wb_reg();
  _leak_reg();
  _na_rtm_reg();
  _na_wb_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
